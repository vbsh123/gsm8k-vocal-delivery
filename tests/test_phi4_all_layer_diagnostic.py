from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_all_layer_diagnostic as run


class AllLayerTests(unittest.TestCase):
    def test_native_generation_recipe_and_observer_parity(self):
        class Batch(dict):
            def to(self,*a,**k):return self
        class Processor:
            def __call__(self,**kwargs):
                return Batch(input_ids=torch.tensor([[200011,200011,7]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                    audio_embed_sizes=torch.tensor([2]))
            def batch_decode(self,*a,**k):return ['#### 2']
        class Block(torch.nn.Module):
            def forward(self,h):return h+1,None
        layers=torch.nn.ModuleList([Block(),Block()]);calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            for h in (torch.zeros(1,3,3),torch.ones(1,1,3)*10):
                for block in layers:h=block(h)[0]
            return torch.tensor([[200011,200011,7,20,21]])
        native=SimpleNamespace(layers=layers,processor=Processor(),config=object(),
            model=SimpleNamespace(device='cpu',generate=generate))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'audio';path.write_bytes(b'audio')
            row=dict(audio_file=str(path),audio_sha256=run.digest(path))
            with patch.object(run.engine,'load_audio',return_value=[]):
                baseline=run.generate(native,row,'Prompt')
                traced=run.generate(native,row,'Prompt',collect=True)
        self.assertEqual(baseline['response'],traced['response'])
        self.assertEqual(traced['means'].shape,(2,2,3));self.assertEqual(traced['hook_forwards'],2)
        torch.testing.assert_close(traced['means'][0,0],torch.full((3,),6.))
        torch.testing.assert_close(traced['means'][0,1],torch.ones(3))
        for call in calls:
            for key,value in dict(max_new_tokens=512,do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True).items():
                self.assertEqual(call[key],value)
        self.assertTrue(all(not layer._forward_hooks for layer in layers))

    def test_collection_and_audio_answer_interventions_preserve_cache_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self, h): return h+1, 'cache'
        layers=torch.nn.ModuleList([Block(),Block()]);mask=torch.tensor([True,True,False])
        prefix=torch.arange(9.).reshape(1,3,3);cached=torch.full((1,1,3),10.)
        with run.hooks(layers,mask,collect=True) as (totals,counts,audio):
            for h in (prefix,cached):
                for block in layers:h=block(h)[0]
        self.assertEqual(counts,{0:2,1:2})
        torch.testing.assert_close(totals[0]/2,(prefix[0,-1]+1+11)/2)
        torch.testing.assert_close(audio[1],prefix[0,:2].mean(0)+2)
        for group in ('audio','answer'):
            original=prefix.clone()
            with run.hooks(layers,mask,target=0,group=group,direction=torch.ones(3)):
                output=layers[0](prefix);second=layers[0](cached)
            self.assertEqual(output[1],'cache');torch.testing.assert_close(prefix,original)
            expected=prefix+1
            if group=='audio':expected[:,mask]-=1
            else:expected[:,-1]-=1
            torch.testing.assert_close(output[0],expected)
            torch.testing.assert_close(second[0],cached+(1 if group=='audio' else 0))
        self.assertTrue(all(not b._forward_hooks for b in layers))
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with run.hooks(layers,mask,collect=True):raise RuntimeError('failure')
        self.assertTrue(all(not b._forward_hooks for b in layers))

    def test_disjoint_collection_reference_selection_and_raw_vector_sign(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'old';source.mkdir();(source/'pairs').mkdir();output=root/'new';output.mkdir()
            questions=[dict(question_id=f'q{i}',question='Q',answer='2') for i in range(5)]
            records=[dict(question_id=q['question_id'],voice='Jenny',style=s,condition=s,reference_answer='2',
                audio_file='audio',audio_sha256=q['question_id']+s) for q in questions for s in ('neutral','angry')]
            mapping={(r['question_id'],r['style']):r for r in records}
            pairs=[dict(question_id=q['question_id'],voice='Jenny',neutral=mapping[q['question_id'],'neutral'],
                angry=mapping[q['question_id'],'angry']) for q in questions[:2]]
            recipe=dict(run.gsm.MODELS['phi4'])
            old=dict(protocol='source',collection=pairs,prompt=run.gsm.ORIGINAL,seed=42,max_new_tokens=512,recipe=recipe,
                source=str(root/'audio'),layer=30)
            config=source/'run_config.json';config.write_text(json.dumps(old));old_hash=run.digest(config)
            for i,pair in enumerate(pairs):
                traces={s:dict(mean=torch.full((3072,),float(j)),hook_forwards=2,response='#### 2',response_tokens=4)
                    for j,s in enumerate(('neutral','angry'))}
                torch.save(dict(protocol='source',run_config_sha256=old_hash,question_id=pair['question_id'],voice='Jenny',
                    audio_sha256={s:pair[s]['audio_sha256'] for s in traces},traces=traces,answer_difference=torch.ones(3072)),source/'pairs'/f'{i:04d}.pt')
            args=SimpleNamespace(collection=source,output=output,neutral_run=root/'neutral',delivery_run=root/'deliveries')
            parent=dict(recipe=recipe,recordings=records)
            with patch.object(run,'TRAIN',2),patch.object(run,'TOTAL',5),patch.object(run.gsm,'questions',return_value=questions),patch.object(run.gsm,'make_plan',return_value=parent):
                plan=run.prepare(args)
            self.assertEqual({r['question_id'] for r in plan['recordings']},{'q2','q3','q4'})
            self.assertFalse({p['question_id'] for p in plan['collection']}&{r['question_id'] for r in plan['recordings']})
            calls=[]
            def generate(native,row,prompt,collect=False):
                calls.append(row)
                return dict(means=torch.full((2,2,3),2. if row['style']=='angry' else 0.),hook_forwards=2,
                    response='#### 2',response_tokens=4)
            with patch.object(run,'SHAPE',(2,2,3)),patch.object(run,'generate',side_effect=generate):
                run.collect_vectors(plan,output,None);run.collect_vectors(plan,output,None)
            self.assertEqual(len(calls),4)
            vectors=torch.load(output/'style-vectors.pt',weights_only=True)
            torch.testing.assert_close(vectors['angry_minus_neutral'],torch.full((2,2,3),2.))
            config_hash=run.digest(output/'run_config.json')
            refs=[]
            for r in plan['recordings']:
                correct=(r['question_id']=='q2' and r['style']=='angry') or r['question_id']=='q3'
                refs.append(dict(run.expected(plan,r),run_config_sha256=config_hash,response='#### 2' if correct else '#### 9',response_tokens=4))
            (output/'references.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
            def score(plan,rows,output):
                return [dict(r,scores={'flexible-numeric':r['response']=='#### 2'}) for r in rows],{'test':True}
            with patch.object(run,'score_rows',side_effect=score):challenge=run.select_challenge(plan,output)
            self.assertEqual(challenge['cases'],[dict(question_id='q2',voice='Jenny')])
            self.assertEqual(challenge['available_test_questions'],3)
            self.assertEqual(json.loads((output/'run_config.json').read_text())['strength'],-1.)


if __name__=='__main__':unittest.main()
