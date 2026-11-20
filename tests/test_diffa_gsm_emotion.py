import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diffa_gsm_emotion as run


class EmotionCollectionTests(unittest.TestCase):
    def test_original_500_and_718_are_disjoint_and_preserved(self):
        train=[f'q{i}' for i in range(500)]
        rows=[dict(question_id=f'q{i}',condition='neutral',voice=run.VOICE) for i in range(500,1218)]
        previous=dict(prompt=run.HASH_PROMPT,recordings=rows)
        a,b=run.select_ids(previous,dict(extraction_question_ids=train))
        self.assertEqual(set(a),set(train));self.assertEqual(b,rows)
        rows[0]['question_id']='q0'
        with self.assertRaisesRegex(ValueError,'disjoint'):run.select_ids(previous,dict(extraction_question_ids=train))

    def test_happy_is_actual_style_and_steering_uses_neutral_audio(self):
        plan=dict(emotion='happy',recordings=[dict(question_id='q',condition=s,style=s,audio_file=s+'.wav') for s in ('neutral','happy')])
        result=run.evaluation_plan(plan,[-1,1,1.5],Path('/tmp/emotion'))
        self.assertEqual(len(result['recordings']),5)
        steered=result['recordings'][2:]
        self.assertEqual([r['strength'] for r in steered],[-1,1,1.5])
        self.assertTrue(all(r['audio_file']=='neutral.wav' and r['style']=='neutral' for r in steered))
        with self.assertRaises(ValueError):run.evaluation_plan(plan,[float('nan')],Path('/tmp'))

    def test_reuse_requires_matching_prompt_audio_seed_and_no_intervention(self):
        plan=dict(prompt=run.HASH_PROMPT,emotion='happy',seed=42,max_new_tokens=512,steps=128,block_length=32,decoder='fast_dllm')
        task=dict(question_id='q',voice=run.VOICE,condition='neutral',audio_sha256='original',answer='5')
        row=dict(task,**{k:plan[k] for k in ('prompt','seed','max_new_tokens','steps','block_length','decoder')},reference_answer='5',response='#### 5',interventions=[])
        self.assertTrue(run.reusable(row,task,plan))
        for changes in (dict(prompt='Final answer: <number>'),dict(audio_sha256='changed'),dict(seed=41),dict(interventions=[dict(strength=1)])):
            self.assertFalse(run.reusable(dict(row,**changes),task,plan))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();out=root/'out';out.mkdir()
            (src/'results.jsonl').write_text(json.dumps(row)+'\n')
            run.reuse(dict(plan,recordings=[task]),out,[src]);run.reuse(dict(plan,recordings=[task]),out,[src])
            self.assertEqual(len(run.read_rows(out/'results.jsonl')),1)

    def test_capture_uses_audio_spans_and_all_forwards_then_removes_hooks(self):
        import torch
        class Block(torch.nn.Module):
            def forward(self,h):return (h,)
        blocks=torch.nn.ModuleList([Block(),Block()])
        model=SimpleNamespace(config=SimpleNamespace(prompt_size=1),llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        original=lambda text,speech,lengths:None
        decoder=SimpleNamespace(insert_audio_embedding=original)
        def generate():
            decoder.insert_audio_embedding(None,torch.zeros(1,3,1),torch.tensor([1]))
            for offset in (0,10):
                h=torch.arange(517,dtype=torch.float32).reshape(1,517,1)+offset
                for block in blocks:h=block(h)[0]
            return '#### 5',dict(mode='native')
        response,_,states=run.capture(model,generate,decoder)
        self.assertEqual(tuple(states.shape),(2,2,4,1))
        self.assertEqual(float(states[0,0,0,0]),1.)
        self.assertEqual(float(states[0,0,1,0]),2.5)
        self.assertEqual(float(states[0,0,2,0]),2.)
        self.assertEqual(float(states[0,0,3,0]),260.5)
        self.assertIs(decoder.insert_audio_embedding,original)
        self.assertTrue(all(not b._forward_hooks for b in blocks))
        n,h,d,k=run.average_pair(states,states[:,:1]+2,'matched')
        self.assertEqual(k,1);self.assertTrue(torch.allclose(d,torch.full_like(d,2.)))

    def test_fit_is_question_weighted_and_emotion_key_is_not_angry(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'pairs').mkdir()
            pairs=[dict(question_id=q) for q in ('a','b')]
            plan=dict(emotion='happy',pairs=pairs,prompt=run.HASH_PROMPT,collection_ids=['a','b'],averaging='matched')
            for i,value in enumerate((2.,6.)):
                torch.save(dict(question_id=pairs[i]['question_id'],difference=torch.full((32,4,2),value)),root/'pairs'/f'{i:05d}.pt')
            with patch.object(run,'analysis'):run.fit(plan,root)
            state=torch.load(root/'style-vectors.pt',weights_only=True)
            self.assertNotIn('angry',state['pair_mean_sums'])
            direction=state['pair_mean_sums']['happy'][31,3]/state['pair_counts']['happy'][31]
            self.assertTrue(torch.allclose(direction,torch.tensor([4.,4.])))
            self.assertEqual(state['extraction_question_ids'],['a','b'])
            before=run.digest(root/'style-vectors.pt')
            run.fit(plan,root)
            self.assertEqual(run.digest(root/'style-vectors.pt'),before)

class EmotionInterventionTests(unittest.TestCase):
    def test_positive_strength_adds_happy_direction_only_to_neutral_answer_positions(self):
        import torch
        class Block(torch.nn.Module):
            def forward(self,h):return (h,)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        model=SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'eval';folder.mkdir()
            sums=torch.zeros(32,4,1);sums[31,3]=8
            torch.save(dict(pair_mean_sums={'happy':sums},pair_counts={'happy':torch.full((32,),2)}),root/'style-vectors.pt')
            plan=dict(emotion='happy',prompt=run.HASH_PROMPT,recordings=[dict(question_id='q',voice=run.VOICE,style=s,condition=s,answer='5') for s in ('neutral','happy')])
            plan=run.evaluation_plan(plan,[1],root)
            calls=[]
            def generate(task,*args):
                h=blocks[31](torch.full((1,514,1),10.))[0]
                calls.append((task['condition'],float(h[0,0,0]),float(h[0,-1,0])))
                return '#### 5',dict(mode='native')
            original=torch.Tensor.to
            def cpu(t,*args,**kw):
                if kw.get('device')=='cuda':kw['device']='cpu'
                return original(t,*args,**kw)
            with patch.object(run,'generate',side_effect=generate),patch.object(torch.Tensor,'to',cpu):
                run.evaluate(plan,folder,root,model,None,None)
                run.evaluate(plan,folder,root,model,None,None)
            self.assertEqual(calls,[('neutral',10.,10.),('happy',10.,10.),('neutral-happy-L31-answer-a1',10.,14.)])
            self.assertTrue(all(not b._forward_hooks for b in blocks))


if __name__=='__main__':unittest.main()
