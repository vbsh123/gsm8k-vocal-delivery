from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import emotion_block_night as night
import tests.test_diffa_vanilla_all as fixtures


class FakeBackend:
    def __init__(self,model):
        self.kind=model
        self.calls=[]
        self.model=self.tokenizer=self.features=None
    def generate(self,record,plan,capture=False):
        style=record['condition'] if 'condition' in record else record['style']
        self.calls.append((record['question_id'],style,capture))
        value={'neutral':1.,'angry':4.,'happy':6.,'sad':-2.}[style]
        trace=dict(response='#### 2',response_tokens=3,audio_encoding={},forwards=1)
        if capture:
            if self.kind=='diffa':
                trace.update(full=torch.full((32,1,4,3),value*100),active=torch.full((32,1,3),value))
            else:
                trace['selected_means']=torch.stack([torch.full((3072,),value*(i+1)) for i in range(3)])
        return trace


class NightTests(unittest.TestCase):
    def registry(self,root,name='gsm8k'):
        return fixtures.AllVanillaTests().registry(root,name)

    def test_selections_are_agreed_only_capped_independent_and_portable(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            registry=self.registry(root/'audio')
            # Only three sad clips qualify: collect all3, instead of cap5. Neutral predictions are all wrong.
            for q in list(registry.neutral_ids('gsm8k'))[3:]:
                registry.labels[q+'sad']['emotion']='other'
            folder,plan,plans=night.prepare('gsm8k',registry,root/'out','diffa',(16,5))
            self.assertEqual([len(plans[e]['collection_ids']) for e in night.EMOTIONS],[5,5,3])
            for e,p in plans.items():
                self.assertEqual(p['collection_ids'],[q for q in plan['question_order'] if q in set(p['collection_ids'])])
                self.assertTrue(all(registry.labels[q+e]['emotion']==e for q in p['collection_ids']))
            self.assertFalse((folder/'random').exists())
            self.assertEqual(len(plan['records']),64)
            _,phi,phi_plans=night.prepare('gsm8k',registry,root/'out','phi',(16,5))
            self.assertEqual(phi['cohort_sha256'],plan['cohort_sha256'])
            self.assertEqual(phi['prompt'],plan['prompt'])
            self.assertEqual(phi['fitting_ids'],plan['fitting_ids'])
            other=self.registry(root/'different-audio-path')
            for q in list(other.neutral_ids('gsm8k'))[3:]:
                other.labels[q+'sad']['emotion']='other'
            m,_=night.cohort('gsm8k',other,(16,5))
            self.assertEqual(night.vanilla.fingerprint(m),plan['cohort_sha256'])
            registry.labels.pop('gsm8k_00happy')
            pending,_=night.cohort('gsm8k',registry,(16,5))
            self.assertIsNone(pending['predictions']['gsm8k_00']['happy'])
            self.assertNotIn('gsm8k_00',pending['fitting_ids']['happy'])
            self.assertEqual(len(pending['question_order']),16)

    def test_all_natural_answers_and_simultaneous_block_fitting_resume_without_repeating_generation(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            folder,plan,plans=night.prepare('gsm8k',self.registry(root/'audio'),root/'out','diffa',(16,5))
            backend=FakeBackend('diffa')
            night.generate_dataset(folder,plan,plans,backend)
            self.assertEqual(len(backend.calls),64)
            self.assertEqual(len({(q,s) for q,s,_ in backend.calls}),64)
            expected_neutral=set().union(*(set(p['collection_ids']) for p in plans.values()))
            self.assertEqual({q for q,s,c in backend.calls if s=='neutral' and c},expected_neutral)
            for e,value in [('angry',3.),('happy',5.),('sad',-3.)]:
                direction,info=night.winners.read_vector(folder/e/'style-vectors.pt',e)
                torch.testing.assert_close(direction,torch.full((3,),value))
                self.assertEqual(info['pairs'],5)
                self.assertEqual({q for q,s,c in backend.calls if s==e and c},set(plans[e]['collection_ids']))
            night.generate_dataset(folder,plan,plans,backend)
            self.assertEqual(len(backend.calls),64)
            self.assertEqual(len(night.raw_rows(folder,plan)),64)
            def generate_job(job,out,*args):
                with (out/'results.jsonl').open('a') as stream:
                    for record in job['target']['records']:
                        row=dict(record,response='#### 2',response_tokens=3,condition='L31-answer-a1',strength=1.,
                            prompt=job['target']['prompt'],seed=42,max_new_tokens=512,steps=128,block_length=32,
                            temperature=0.,decoder='fast_dllm',vector_sha256=job['source']['vector_sha256'],
                            interventions=[dict(layer=31,group='answer',strength=1.)],steered_forwards=1)
                        stream.write(json.dumps(row)+'\n')
            with patch.object(night.winners,'generate_job',side_effect=generate_job),patch.object(night.vanilla.task,'report_steering'):
                night.steer_dataset(folder,plan,plans,backend)
            for e in night.EMOTIONS:
                job=json.loads((folder/e/'L31-answer-a1/run_config.json').read_text())
                self.assertEqual(len(job['target']['records']),11)
                self.assertFalse(set(plans[e]['collection_ids']) & {r['question_id'] for r in job['target']['records']})

    def test_resume_after_partial_question_does_not_duplicate_deliveries_or_pairs(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            folder,plan,plans=night.prepare('gsm8k',self.registry(root/'audio'),root/'out','diffa',(16,5))
            backend=FakeBackend('diffa')
            original=night.append_pair
            count=[0]
            def interrupt(*args):
                count[0]+=1
                if count[0]==2:
                    raise RuntimeError('simulated interruption after raw answer')
                return original(*args)
            with patch.object(night,'append_pair',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    night.generate_dataset(folder,plan,plans,backend)
            night.generate_dataset(folder,plan,plans,backend)
            self.assertEqual(len(night.raw_rows(folder,plan)),64)
            for e,p in plans.items():
                state=night.load_state(p,folder/e)
                self.assertEqual(state['completed_ids'],p['collection_ids'])
                self.assertEqual(len(state['rows']),5)
            self.assertGreater(len(backend.calls),64)

    def test_phi_collects_all_three_directions_for_every_emotion_in_same_pass(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            folder,plan,plans=night.prepare('gsm8k',self.registry(root/'audio'),root/'out','phi',(16,5))
            backend=FakeBackend('phi')
            night.generate_dataset(folder,plan,plans,backend)
            night.generate_dataset(folder,plan,plans,backend)
            self.assertEqual(len(backend.calls),64)
            for e,value in [('angry',3.),('happy',5.),('sad',-3.)]:
                vector=torch.load(folder/e/'style-vectors.pt',weights_only=True)
                self.assertEqual(set(vector['directions']),set(night.PHI_TARGETS))
                self.assertEqual(vector['collection_question_ids'],plans[e]['collection_ids'])
                for index,key in enumerate(night.PHI_TARGETS):
                    torch.testing.assert_close(vector['directions'][key],torch.full((3072,),value*(index+1)))
            with self.assertRaisesRegex(ValueError,'no steering'):
                night.steer_dataset(folder,plan,plans,backend)

    def test_native_phi_wrapper_chooses_audio0_and_answer11_30(self):
        import phi4_all_layer_diagnostic as hooks
        backend=night.PhiBackend.__new__(night.PhiBackend)
        backend.native=object()
        values=torch.arange(6.).reshape(3,2,1).expand(3,2,3072)
        trace=dict(response='answer',response_tokens=1,means=values,hook_forwards=7)
        with patch.object(hooks,'generate',return_value=trace) as generate:
            output=backend.generate({},dict(prompt='bare',max_new_tokens=512),capture=True)
        self.assertEqual(generate.call_args.kwargs['capture_layers'],[0,11,30])
        torch.testing.assert_close(output['selected_means'][:,0],torch.tensor([1.,2.,4.]))

    def test_official_score_entrypoint_is_identical_for_models_and_all_pairs_reported(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            for model in ('diffa','phi'):
                folder,plan,plans=night.prepare('gsm8k',self.registry(root/model),root/'out',model,(16,5))
                night.generate_dataset(folder,plan,plans,FakeBackend(model))
                def score(config,rows,path):
                    self.assertEqual(config['target']['prompt'],night.catalog.prompt_for('gsm8k'))
                    return [dict(r,metrics={'flexible-numeric':True}) for r in rows],{'official':'same scorer'}
                with patch.object(night.winners,'score_rows',side_effect=score):
                    night.natural_report(folder,plan)
                report=json.loads((folder/'delivery-summary.json').read_text())
                self.assertEqual(len(report['comparisons']),6)
                self.assertTrue(all(c['pairs']==16 for c in report['comparisons']))
                self.assertEqual(report['scorer'],{'official':'same scorer'})

    def test_machine_wrapper_runs_exact_dataset_order_then_switches_phi_environment(self):
        import os
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            script=root/'run_emotion_block_night.sh'
            shutil.copyfile(Path(night.__file__).with_name(script.name),script)
            for envname in ('.venv-diffa','.venv-bbh-phi4'):
                binary=root/envname/'bin/python'
                binary.parent.mkdir(parents=True)
                binary.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["EMOTION_NIGHT_TEST_LOG"],"a") as f: f.write(json.dumps(sys.argv)+"\\n")\n')
                binary.chmod(0o755)
            log=root/'calls.jsonl'
            env=dict(os.environ,EMOTION_NIGHT_TEST_LOG=str(log),DIFFA_PYTHON=str(root/'.venv-diffa/bin/python'))
            subprocess.run(['bash',str(script),'phi'],env=env,check=True,timeout=10)
            calls=[json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls),4)
            self.assertEqual([row[row.index('--model')+1] for row in calls],['diffa','phi','diffa','phi'])
            self.assertEqual([row[row.index('--mode')+1] for row in calls],['prepare','prepare','run','run'])
            for row in (calls[0],calls[2]):
                self.assertEqual(row[row.index('--datasets')+1:row.index('--mode')],['svamp','ifeval'])
            self.assertIn('.venv-bbh-phi4',calls[3][0])
            self.assertEqual(calls[3][calls[3].index('--datasets')+1],'gsm8k')
            for role,datasets in [('old',['gsm8k','multiarith']),('new',['bbh3','bbh5'])]:
                log.write_text('')
                subprocess.run(['bash',str(script),role,'--mode','prepare'],env=env,check=True,timeout=10)
                row=json.loads(log.read_text())
                self.assertEqual(row[row.index('--datasets')+1:row.index('--mode')],datasets)

    def test_ifeval_reports_official_prompt_and_instruction_metrics(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            folder,plan,plans=night.prepare('ifeval',self.registry(root/'audio','ifeval'),root/'out','diffa',(16,5))
            self.assertEqual(plan['prompt'],'Follow the instructions in the audio.')
            self.assertEqual(plan['max_new_tokens'],1024)
            night.generate_dataset(folder,plan,plans,FakeBackend('diffa'))
            def score(config,rows,path):
                return [dict(r,metrics=dict(strict_prompt=False,loose_prompt=True),official_scores={
                    'strict':dict(instructions=[True,False]),'loose':dict(instructions=[True,True])}) for r in rows],{}
            with patch.object(night.winners,'score_rows',side_effect=score):
                night.natural_report(folder,plan)
            report=json.loads((folder/'delivery-summary.json').read_text())
            for s in night.STYLES:
                self.assertEqual(report['instruction_totals'][s]['strict']['accuracy'],0.5)
                self.assertEqual(report['instruction_totals'][s]['loose']['accuracy'],1.0)
