import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_step_by_step_refit as refit
from evaluate_gsm_candidates import EXPLICIT,ORIGINAL


class StepByStepRefitTests(unittest.TestCase):
    def fixture(self,root,cohort='original',name='gsm8k'):
        def record(q,s):
            return dict(question_id=q,voice='Jenny',style=s,audio_sha256=q+s,audio_file='/audio/'+q+s,
                        reference_answer='2',seconds=1.)
        train,test=['q0','q1'],['q2','q3']
        bank={q:dict(question_id=q,question='Question '+q,answer='2') for q in train+test}
        target=dict(name=name,task=name,records=[record(q,'neutral') for q in test],bank={q:bank[q] for q in test},
                    prompt=EXPLICIT,seed=42,max_new_tokens=512,steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                    vector='/missing/original-vector.pt',vector_info={},reuse_runs=[])
        sources={}
        for e in refit.EMOTIONS:
            style=e if cohort=='shared3' else 'happy'
            plan=dict(name=name,emotion=style,seed=42,collection_ids=train,test_ids=test,bank=bank,target=target,
                      max_new_tokens=512,prompt=EXPLICIT,
                      collection_prompt=EXPLICIT.replace('#### <number>','Final answer: <number>') if cohort=='original' else EXPLICIT,
                      pairs=[dict(question_id=q,neutral=record(q,'neutral'),emotion=record(q,style)) for q in train],
                      recordings=[dict(record(q,s),condition=s) for q in test for s in ('neutral',style)])
            path=root/f'{cohort}-{name}-{e}.json';path.write_text(json.dumps(plan));sources[e]=path
        def choose(dataset,q,style,**kwargs):
            self.assertEqual(kwargs['question'],bank[q]['question'])
            self.assertEqual(kwargs['answer'],'2')
            if kwargs.get('expected'):self.assertEqual(kwargs['expected'],q+style)
            return dict(record=record(q,style))
        return sources,SimpleNamespace(choose=choose,options={})

    def test_original_and_shared3_keep_splits_and_format_but_retain_step_by_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in refit.DATASETS:
                for cohort in refit.COHORTS:
                    sources,registry=self.fixture(root,cohort,name)
                    plans=refit.prepare(name,cohort,sources,registry,root/cohort/name)
                    for e,p in plans.items():
                        self.assertEqual(p['prompt'],ORIGINAL)
                        self.assertIn("Let's think step by step.",p['collection_prompt'])
                        self.assertNotIn('necessary calculations',p['collection_prompt'])
                        self.assertEqual(p['collection_ids'],['q0','q1'])
                        self.assertEqual(p['test_ids'],['q2','q3'])
                        self.assertEqual({r['condition'] for r in p['recordings']},{'neutral',e})
                        if cohort=='original':self.assertIn('Final answer: <number>',p['collection_prompt'])
                        else:self.assertIn('#### <number>',p['collection_prompt'])
            with self.assertRaises(ValueError):refit.step_prompt(ORIGINAL)

    def test_triad_collection_uses_neutral_once_and_resumes_partial_emotion_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sources,registry=self.fixture(root)
            output=root/'out';plans=refit.prepare('gsm8k','original',sources,registry,output)
            seen=[];fail=[True]
            def generate(record,*args):
                seen.append((record['question_id'],record['style']))
                return record['style'],{}
            def capture(model,fn,decoder,budget):
                style,encoding=fn()
                if style=='happy' and fail[0]:
                    fail[0]=False
                    raise RuntimeError('interrupted')
                value={'neutral':1.,'angry':3.,'happy':5.}[style]
                return style,encoding,torch.full((32,2,4,3),value)
            src=ModuleType('src');src.__path__=[]
            fast=ModuleType('src.fast_dllm');fast.__path__=[]
            decoder=ModuleType('src.fast_dllm.diffa_generate')
            with patch.dict(sys.modules,{'src':src,'src.fast_dllm':fast,'src.fast_dllm.diffa_generate':decoder}),patch.object(
                    refit.task,'generate',side_effect=generate),patch.object(refit.task.emotion,'capture',side_effect=capture):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    refit.collect_both(plans,output,None,None,None)
                refit.collect_both(plans,output,None,None,None)
                done=len(seen)
                refit.collect_both(plans,output,None,None,None)
                self.assertEqual(len(seen),done)
            self.assertEqual(seen.count(('q0','angry')),1)
            self.assertEqual(seen.count(('q1','neutral')),1)
            for e,delta in [('angry',2.),('happy',4.)]:
                vector,info=refit.task.winners.read_vector(output/e/'style-vectors.pt',e)
                torch.testing.assert_close(vector,torch.full((3,),delta))
                self.assertEqual(info['extraction_ids'],['q0','q1'])
                self.assertEqual(info['pairs'],2)

    def test_delivery_reuse_rejects_calculation_prompt_and_hidden_nonzero_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sources,registry=self.fixture(root)
            output=root/'out';plan=refit.prepare('gsm8k','original',sources,registry,output)['angry']
            records={r['question_id']:r for r in plan['recordings'] if r['condition']=='neutral'}
            metadata={k:plan['target'][k] for k in ('seed','max_new_tokens','steps','block_length','temperature','decoder')}
            def row(q,prompt,response,strength=0.):
                return dict(records[q],response=response,prompt=prompt,response_tokens=2,strength=strength,
                            interventions=[],**metadata)
            source=root/'source-delivery';source.mkdir()
            data=[row('q2',EXPLICIT,'wrong prompt'),row('q2',ORIGINAL,'correct'),row('q3',ORIGINAL,'invalid strength',1.)]
            (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
            delivery=output/'angry/delivery'
            refit.reuse_deliveries(plan,delivery,[source])
            saved=list(refit.deliveries(plan,delivery).values())
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0]['response'],'correct')


if __name__=='__main__':unittest.main()
