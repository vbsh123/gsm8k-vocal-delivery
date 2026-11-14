from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from evaluate_arc_audio import PROMPT,STEERING,main,report,run,score
from generate_arc_azure import select_questions


class ArcTests(unittest.TestCase):
    def row(self,i=0):
        return dict(id=f'q{i}',question='What dissolves table salt?',choices=dict(label=['1','2','3','4'],text=['water','oil','sand','air']),answerKey='1')

    def test_fixed_sampling_preserves_choice_order_and_remaps_gold(self):
        rows=[self.row(i) for i in range(10)]
        selected,excluded=select_questions(rows,5)
        self.assertEqual((selected,excluded),select_questions(rows,5))
        self.assertEqual(len(selected),5);self.assertFalse(excluded)
        for row in selected:
            self.assertEqual(row['answer'],'A');self.assertEqual(row['original_answer'],'1')
            self.assertEqual(row['choices'],[dict(label=k,text=v) for k,v in zip('ABCD',['water','oil','sand','air'])])
            self.assertNotIn('answerKey',row['question'])

    def test_visual_math_markup_and_nonfour_choices_excluded_before_sampling(self):
        good=self.row();diagram=dict(self.row(1),question='Look at the diagram below.')
        chemistry=dict(self.row(2),question='What is CO_{2}?')
        few=dict(self.row(3),choices=dict(label=['A','B'],text=['x','y']),answerKey='A')
        selected,excluded=select_questions([diagram,good,chemistry,few],1)
        self.assertEqual(selected[0]['source_id'],'q0');self.assertEqual(len(excluded),3)

    def test_answer_correctness_is_separate_from_format_compliance(self):
        q=select_questions([self.row()],1)[0][0]
        a=score('A',q);b=score('Final answer: A',q)
        self.assertTrue(a['answer_correct']);self.assertFalse(a['format_compliant']);self.assertFalse(a['correct'])
        self.assertTrue(b['answer_correct']);self.assertTrue(b['format_compliant']);self.assertTrue(b['correct'])
        invalid=score('Final answer: E',q)
        self.assertFalse(invalid['format_compliant']);self.assertTrue(invalid['answer_parse_failed'])

    def test_prepare_freezes_five_trials_without_model_loading(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'audio';source.mkdir();output=root/'out';vector=root/'vector.pt';vector.write_bytes(b'fixed-vector')
            q=select_questions([self.row()],1)[0][0];rows=[]
            (source/'questions.jsonl').write_text(json.dumps(q)+'\n')
            for style in ('neutral','angry','happy','sad'):
                with wave.open(str(source/(style+'.wav')),'wb') as a:
                    a.setnchannels(1);a.setsampwidth(2);a.setframerate(16000);a.writeframes(b'\0\0'*160)
                rows.append(dict(question_id=q['question_id'],provider='azure',voice='Jenny',style=style,
                    audio_path=style+'.wav',answer=q['answer'],question=q['question']))
            (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=['test','--source',str(source),'--output',str(output),'--vectors',str(vector),'--prepare-only']
            with patch('sys.argv',args):main()
            config=json.loads((output/'run_config.json').read_text())
            self.assertEqual(len(config['recordings']),5);self.assertEqual(config['seed'],42)
            self.assertEqual(config['decoder'],'fast_dllm');self.assertEqual(config['prompt'],PROMPT)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Needs NumPy for paired intervals')
    def test_paired_report_does_not_call_a_formatting_gain_an_answer_gain(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);q=select_questions([self.row()],1)[0][0]
            (root/'questions.jsonl').write_text(json.dumps(q)+'\n')
            rows=[dict(question_id=q['question_id'],voice='Jenny',reference_answer='A',condition=c,response=r)
                for c,r in [('neutral','A'),(STEERING,'Final answer: A')]]
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));report(root,root)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual([r['gain_pp'] for r in summary['paired']],[0.,100.])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class ArcRuntimeTests(unittest.TestCase):
    def test_direction_controls_prepare_both_cohorts_with_reused_baselines(self):
        import torch
        from compare_cloud_accuracy import PROMPT as GSM_PROMPT
        from run_L31_direction_controls import main as prepare,METHODS
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);vector=root/'original.pt';late=root/'late';out=root/'out'
            sums=torch.zeros(32,4,3);sums[31,3]=torch.tensor([1.,2.,3.]);counts=torch.ones(32,dtype=torch.long)
            torch.save(dict(prompt=GSM_PROMPT,max_new_tokens=512,groups=['acoustic','semantic','combined','answer'],
                pair_mean_sums={'angry':sums},pair_counts={'angry':counts}),vector)
            for i,folder in enumerate([late/'original_mean',late/'third-cohort/original_mean']):
                folder.mkdir(parents=True)
                (folder/'run_config.json').write_text(json.dumps(dict(settings=dict(vectors=str(vector)),recordings=[dict(question_id=f'gsm8k_test_{i:05d}')])) )
            with patch('sys.argv',['test','--late-root',str(late),'--output',str(out),'--prepare-only']),patch('run_L31_direction_controls.subprocess.run') as called:
                prepare()
            self.assertEqual(called.call_count,2)
            for call in called.call_args_list:
                command=call.args[0]
                self.assertIn('--baseline-source',command);self.assertEqual(command[-1],'--prepare-only')
                self.assertEqual(command[command.index('--methods')+1:command.index('--conditions-file')],METHODS)
            for method in METHODS:
                state=torch.load(out/'vectors'/method/'style-vectors.pt',weights_only=True)
                self.assertEqual(int(state['pair_counts']['angry'][31]),1)
                self.assertAlmostEqual(float(state['pair_mean_sums']['angry'][31,3].norm()),float(sums[31,3].norm()),places=6)

    def test_random_controls_match_norm_are_deterministic_and_leave_global_rng_alone(self):
        import torch
        from run_L31_direction_controls import control_directions
        vector=torch.arange(1,4097,dtype=torch.float32)
        torch.manual_seed(42);before=torch.random.get_rng_state().clone()
        controls=control_directions(vector,torch)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(controls['reverse'],-vector))
        for name,control in controls.items():
            self.assertAlmostEqual(float(control.norm()/vector.norm()),1.,places=6)
            self.assertTrue(torch.equal(control,control_directions(vector,torch)[name]))
        self.assertFalse(torch.equal(controls['random1'],controls['random2']))

    def test_transfer_uses_frozen_mean_seed42_and_resumes(self):
        import numpy as np
        import torch
        class Block(torch.nn.Module):
            def forward(self,h):return (h,None)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        model=SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        calls=[]
        def generate(audio,sr,model,tokenizer,features,prompt,limit):
            h=torch.zeros(1,520,3)
            for layer in blocks:h=layer(h)[0]
            calls.append((float(h[0,-1,0]),float(h[0,0,0]),float(torch.rand(())),prompt,limit))
            return 'Final answer: A'
        original_to=torch.Tensor.to
        def cpu_to(tensor,*a,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return original_to(tensor,*a,**kw)
        modules={'soundfile':SimpleNamespace(read=lambda *a,**kw:(np.zeros(20,dtype=np.float32),16000)),
            'extract_vectors':SimpleNamespace(load_model=lambda:(model,None,None),gen=generate)}
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);vector=root/'vectors.pt';audio=root/'audio.wav';audio.write_bytes(b'audio');stat=audio.stat()
            sums=torch.zeros(32,4,3);sums[31,3]=2;counts=torch.zeros(32,dtype=torch.long);counts[31]=2
            torch.save(dict(groups=['acoustic','semantic','combined','answer'],pair_mean_sums={'angry':sums},pair_counts={'angry':counts},extraction_question_ids=['cs-training']),vector)
            q=select_questions([ArcTests().row()],1)[0][0]
            tasks=[dict(question_id=q['question_id'],voice='Jenny',style='neutral',audio_path='audio.wav',
                audio_file=str(audio),audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns,condition=c) for c in ('neutral',STEERING)]
            config=dict(vector=dict(path=str(vector)),recordings=tasks)
            with patch.dict('sys.modules',modules),patch.object(torch.Tensor,'to',cpu_to):
                run(config,{q['question_id']:q},root);run(config,{q['question_id']:q},root)
            self.assertEqual(len(calls),2);self.assertEqual([c[:2] for c in calls],[(0.,0.),(1.,0.)])
            self.assertEqual(calls[0][2],calls[1][2]);self.assertEqual(calls[0][3:],(PROMPT,512))
            self.assertTrue(all(not b._forward_hooks for b in blocks))


if __name__=='__main__':unittest.main()
