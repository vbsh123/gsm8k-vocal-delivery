import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import torch
import diffa_happy_nightly as run


class HappyNightlyTests(unittest.TestCase):
    def test_default_queue_is_only_collection_delivery_and_own_plus1(self):
        args=run.arguments([])
        self.assertEqual(args.stages,['primary'])
        self.assertEqual(args.strengths,[1.])
        self.assertEqual(args.datasets,list(run.DATASETS))

    def test_audio_inventory_skips_unrelated_legacy_entries_without_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);clip=root/'clip.wav';clip.write_bytes(b'audio')
            valid=dict(question_id='q',question='question',audio_path='clip.wav',provider='azure',voice=run.VOICE,style='happy')
            invalid=[None,'legacy',dict(valid,question_id=None),{k:v for k,v in valid.items() if k!='question_id'},
                {k:v for k,v in valid.items() if k!='audio_path'},dict(valid,voice='Ryan'),dict(valid,audio_path='missing.wav')]
            (root/'plan.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in [*invalid,valid]))
            found=run.audio_inventory(root)
            self.assertEqual(list(found),[('q','happy')])
            self.assertEqual(found['q','happy'][0][0],valid)

    def fixture(self,root):
        audio=root/'audio';audio.mkdir();out=root/'new';out.mkdir();old=root/'old';old.mkdir()
        inventory={}
        for qid in ('train','test'):
            for style in ('neutral','happy'):
                path=audio/f'{qid}-{style}.wav'
                with wave.open(str(path),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*16)
                row=dict(question_id=qid,voice=run.VOICE,provider='azure',style=style,question='question '+qid,answer='2',audio_path=path.name)
                inventory[qid,style]=[(row,audio)]
        row,folder=inventory['test','neutral'][0];record=run.make_record(row,folder)
        config=dict(seed=42,decoder='fast_dllm',steps=128,block_length=32,temperature=0.,prompt='eval prompt',max_new_tokens=1024,recordings=[dict(record,condition='neutral')])
        (old/'run_config.json').write_text(json.dumps(config))
        saved=dict(record,response='#### 2',condition='neutral',prompt='eval prompt',seed=42,max_new_tokens=1024,decoder='fast_dllm',steps=128,block_length=32,temperature=0.,interventions=[])
        (old/'results.jsonl').write_text(json.dumps(saved)+'\n')
        target=dict(name='ifeval',task='ifeval',run=str(old.absolute()),config_file=str(old/'run_config.json'),config_sha256=run.winners.digest(old/'run_config.json'),
            vector_sha256='oldvector',vector_info=dict(collection_prompt='collection prompt',width=3),extraction_ids=['train'],records=[record],
            bank={'test':dict(question='question test',answer='2')},prompt='eval prompt',max_new_tokens=1024,reuse_runs=[],seed=42,decoder='fast_dllm',steps=128,block_length=32,temperature=0.)
        return out,old,inventory,target

    def test_plan_preserves_collection_ids_prompts_budget_and_neutral_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,old,inventory,target=self.fixture(root)
            with patch.object(run.winners,'load_dataset',return_value=target):
                plan=run.task_plan('ifeval',dict(run=str(old)),out,inventory)
            self.assertEqual(plan['collection_ids'],['train']);self.assertEqual(plan['test_ids'],['test'])
            self.assertEqual(plan['collection_prompt'],'collection prompt');self.assertEqual(plan['prompt'],'eval prompt')
            self.assertEqual(plan['max_new_tokens'],1024)
            self.assertEqual(plan['recordings'][0]['audio_sha256'],target['records'][0]['audio_sha256'])

    def test_heldout_neutral_is_reused_and_only_happy_is_generated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,old,inventory,target=self.fixture(root)
            with patch.object(run.winners,'load_dataset',return_value=target):plan=run.task_plan('ifeval',dict(run=str(old)),out,inventory)
            delivery=out/'ifeval'/'delivery';delivery.mkdir()
            run.reuse_delivery(plan,delivery)
            tokenizer=SimpleNamespace(encode=lambda text,**kwargs:[1,2])
            with patch.object(run,'generate',return_value=('#### 2',dict(mode='native'))) as generate:
                run.deliver(plan,delivery,None,tokenizer,None);run.deliver(plan,delivery,None,tokenizer,None)
            self.assertEqual(generate.call_count,1);self.assertEqual(generate.call_args.args[0]['style'],'happy')
            self.assertEqual(len(run.read_rows(delivery/'results.jsonl')),2)
            (old/'results.jsonl').unlink();(delivery/'results.jsonl').unlink()
            with self.assertRaisesRegex(ValueError,'no neutral evaluation'):run.reuse_delivery(plan,delivery)

    def test_only_missing_happy_audio_is_generated(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,old,inventory,target=self.fixture(root)
            original,source=inventory.pop(('train','happy'))[0]
            calls=[]
            def tts(destination,questions,styles):
                calls.append((questions,styles));destination.mkdir(parents=True)
                shutil.copyfile(source/original['audio_path'],destination/'generated.wav')
                row=dict(original,audio_path='generated.wav')
                (destination/'plan.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(run.winners,'load_dataset',return_value=target),patch('run_reasoning_transfer.generate_audio',side_effect=tts):
                plan=run.task_plan('ifeval',dict(run=str(old)),out,inventory)
            self.assertEqual(len(calls),1)
            self.assertEqual([q['question_id'] for q in calls[0][0]],['train'])
            self.assertEqual(calls[0][1],['happy'])
            self.assertEqual(plan['pairs'][0]['emotion']['style'],'happy')
            self.assertEqual(plan['recordings'][0]['audio_sha256'],target['records'][0]['audio_sha256'])

    def test_happy_vector_key_and_positive_sign(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'vectors.pt';sums=torch.zeros(32,4,3);sums[31,3]=6
            state=dict(pair_mean_sums={'happy':sums},pair_counts={'happy':torch.full((32,),2)},groups=run.GROUPS,extraction_question_ids=['train'],prompt='collection',max_new_tokens=1024)
            torch.save(state,path)
            vector,info=run.winners.read_vector(path,'happy')
            torch.testing.assert_close(vector,torch.full((3,),3.));self.assertEqual(info['collection_tokens'],1024)
            with self.assertRaises(ValueError):run.winners.read_vector(path)

    def test_transfer_preserves_target_prompt_and_rejects_leakage(self):
        source=dict(extraction_ids=['train'],emotion='happy');target=dict(records=[dict(question_id='test')],prompt='target',max_new_tokens=1024)
        job=run.steering_plan(source,target,[0.,1.])
        self.assertEqual(job['source']['emotion'],'happy');self.assertEqual(job['target']['max_new_tokens'],1024)
        with self.assertRaises(ValueError):run.steering_plan(dict(source,extraction_ids=['test']),target,[1.])

    def test_capture_respects1024_answer_positions(self):
        class Block(torch.nn.Module):
            def forward(self,h):return (h,)
        blocks=torch.nn.ModuleList([Block()]);model=SimpleNamespace(config=SimpleNamespace(prompt_size=1),llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        original=lambda *args:None;decoder=SimpleNamespace(insert_audio_embedding=original)
        def generate():
            decoder.insert_audio_embedding(None,torch.zeros(1,3,1),torch.tensor([1]));blocks[0](torch.arange(1029.).view(1,1029,1));return 'answer',{}
        _,_,captured=run.emotion.capture(model,generate,decoder,1024)
        self.assertEqual(float(captured[0,0,3,0]),516.5)
        self.assertIs(decoder.insert_audio_embedding,original);self.assertFalse(blocks[0]._forward_hooks)

    def test_actual_contributing_pairs_are_not_the_larger_exclusion_list(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.pt'
            torch.save(dict(pairs=[dict(question_id='selected')]),path)
            target=dict(extraction_ids=['selected','excluded'],vector_info=dict(pairs=1),vector=str(path))
            self.assertEqual(run.historical_collection_ids(target),['selected'])
            target['vector_info']['pairs']=3
            with self.assertRaises(ValueError):run.historical_collection_ids(target)

    def test_native_generation_reads_happy_vector_and_adds_positive_answer_shift(self):
        import sys
        class Block(torch.nn.Module):
            def forward(self,h):return (h,)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        model=SimpleNamespace(llm_model=SimpleNamespace(get_input_embeddings=lambda:torch.nn.Embedding(1,3),
            model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,old,inventory,target=self.fixture(root)
            row,source=inventory['test','neutral'][0];record=run.make_record(row,source)
            vectors=root/'happy.pt';sums=torch.zeros(32,4,3);sums[31,3]=6
            torch.save(dict(pair_mean_sums={'happy':sums},pair_counts={'happy':torch.full((32,),2)},groups=run.GROUPS),vectors)
            origin=dict(name='source',emotion='happy',vector=str(vectors),vector_sha256=run.winners.digest(vectors))
            target=dict(name='target',records=[record],prompt='eval',max_new_tokens=512)
            plan=dict(source=origin,target=target,strengths=[1.])
            observed=[]
            def gen(*args):
                h=blocks[31](torch.full((1,514,3),10.))[0]
                observed.append((float(h[0,0,0]),float(h[0,-1,0])))
                return '#### 2',dict(mode='native')
            original=torch.Tensor.to
            def cpu(t,*args,**kw):
                if kw.get('device')=='cuda':kw['device']='cpu'
                return original(t,*args,**kw)
            folder=root/'steer';folder.mkdir()
            with patch.dict(sys.modules,{'extract_vectors':SimpleNamespace(gen_chunked=gen)}),patch.object(torch.Tensor,'to',cpu):
                run.winners.generate_job(plan,folder,model,None,None)
                run.winners.generate_job(plan,folder,model,None,None)
            self.assertEqual(observed,[(10.,13.)])
            self.assertEqual(len(run.read_rows(folder/'results.jsonl')),1)
            self.assertTrue(all(not block._forward_hooks for block in blocks))

    def test_nightly_finishes_primary_tasks_before_strengths_and_transfer(self):
        import os,sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);catalog={};targets={};inventory={}
            for name in ('gsm8k','svamp'):
                base=root/name;base.mkdir()
                _,old,found,target=self.fixture(base)
                target.update(name=name,task=name)
                targets[str(old)]=target;catalog[name]=dict(run=str(old))
                inventory.update(found)
            path=root/'catalog.json';path.write_text(json.dumps(catalog))
            output=root/'night';events=[]
            def collect(plan,folder,*args):
                events.append(('collect',plan['name']))
                torch.save(dict(pair_mean_sums={'happy':torch.ones(32,4,3)},pair_counts={'happy':torch.ones(32,dtype=torch.long)},
                    groups=run.GROUPS,prompt=plan['collection_prompt'],max_new_tokens=plan['max_new_tokens']),folder/'style-vectors.pt')
            def generation(job,*args):events.append(('steer',job['source']['name'],job['target']['name'],job['strengths']))
            model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda:{}))
            previous=Path.cwd();os.chdir(root);(root/'outputs').mkdir()
            try:
                with patch.object(sys,'argv',['night','--catalog',str(path),'--output',str(output),'--datasets','gsm8k','svamp','--stages','primary','strengths','transfer','--strengths','-1','1']), \
                    patch.object(run,'audio_inventory',return_value=inventory), \
                    patch.object(run.winners,'load_dataset',side_effect=lambda name,entry:targets[entry['run']]), \
                    patch.object(run,'collect',side_effect=collect),patch.object(run,'deliver'), \
                    patch.object(run,'report_delivery'),patch.object(run,'report_steering'), \
                    patch.object(run.winners,'score_rows',return_value=([],{})), \
                    patch.object(run.winners,'generate_job',side_effect=generation), \
                    patch.object(run.subprocess,'check_output',return_value=''), \
                    patch.dict(sys.modules,{'extract_vectors':SimpleNamespace(load_model=lambda:(model,None,None))}):
                    run.main()
            finally:os.chdir(previous)
            self.assertEqual(events[:4],[('collect','gsm8k'),('steer','gsm8k','gsm8k',[0.,1.]),('collect','svamp'),('steer','svamp','svamp',[0.,1.])])
            self.assertEqual(events[-2:],[('steer','gsm8k','svamp',[0.,1.]),('steer','svamp','gsm8k',[0.,1.])])
            self.assertTrue((output/'queue_manifest.json').is_file())

if __name__=='__main__':unittest.main()
