from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_active_pooling as active
import tests.test_diffa_layer_probe as probe_tests


class ActivePoolingTests(unittest.TestCase):
    def test_capture_tracks_current_block_and_preserves_native_forwards(self):
        class Block(torch.nn.Module):
            def forward(self, h):
                return h+1, None
        class LLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(126337,1)
                self.model=SimpleNamespace(transformer=SimpleNamespace(blocks=
                    torch.nn.ModuleList([Block() for _ in range(32)])))
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, h):
                for block in self.model.transformer.blocks:
                    h=block(h)[0]
                return h
        llm=LLM()
        model=SimpleNamespace(config=SimpleNamespace(prompt_size=2),llm_model=llm)
        decoder=SimpleNamespace(insert_audio_embedding=lambda t,s,l:torch.cat([t[:,:1],s,t[:,1:]],dim=1))
        original=decoder.insert_audio_embedding
        observed=[]
        def generate():
            # Native embeddings see the full token state before each decoder forward.
            ids=torch.full((1,10),126336,dtype=torch.long)
            ids[:,:2]=2
            llm.embedding(ids)
            h=decoder.insert_audio_embedding(torch.zeros(1,10,1),torch.zeros(1,4,1),torch.tensor([1]))
            h[:,-8:-4]=10
            h[:,-4:]=100
            observed.append(llm(h).clone())
            ids[:,-8:-4]=2  # First block complete: second block is now current.
            llm.embedding(ids)
            observed.append(llm(h).clone())
            return '#### 2',{}
        with torch.inference_mode():
            response,encoding,full,current=active.capture_poolings(model,generate,decoder,8,4)
        self.assertEqual(response,'#### 2')
        torch.testing.assert_close(full[0,:,3,0],torch.tensor([56.,56.]))
        torch.testing.assert_close(current[0,:,0],torch.tensor([11.,101.]))
        torch.testing.assert_close(current[31,:,0],torch.tensor([42.,132.]))
        torch.testing.assert_close(observed[0],observed[1])
        self.assertIs(decoder.insert_audio_embedding,original)
        self.assertFalse(llm._forward_pre_hooks)
        self.assertFalse(llm.embedding._forward_pre_hooks)
        self.assertTrue(all(not b._forward_hooks for b in llm.model.transformer.blocks))
        def fail():
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            active.capture_poolings(model,fail,decoder,8,4)
        self.assertFalse(llm._forward_pre_hooks)
        self.assertFalse(llm.embedding._forward_pre_hooks)
        self.assertTrue(all(not b._forward_hooks for b in llm.model.transformer.blocks))

    def test_pooling_changes_only_answer_positions_not_time_weighting(self):
        n=torch.ones(32,2,4,3)
        e=torch.ones(32,5,4,3)*3
        na=torch.ones(32,2,3)*7
        ea=torch.ones(32,5,3)*12
        ea[:,2:]=1000  # Excluded by exactly the same matched-forward rule.
        pooled,count=active.pool_pair(n,e,na,ea)
        self.assertEqual(count,2)
        torch.testing.assert_close(pooled['full-buffer'][2],torch.full((32,4,3),2.))
        torch.testing.assert_close(pooled['active-block'][2][:,3],torch.full((32,3),5.))
        torch.testing.assert_close(pooled['active-block'][2][:,:3],torch.full((32,3,3),2.))

    def fixture(self,root):
        bundle,_=probe_tests.LayerProbeTests().fixture(root,'agreed')
        source=root/'outputs/diffa-gsm-vanilla/angry/agreed'
        vector=torch.load(source/'style-vectors.pt',weights_only=True)
        plan=dict(protocol='test',emotion='angry',regime='agreed',prompt=active.vanilla.PROMPT,
            collection_prompt=active.vanilla.PROMPT,**active.vanilla.RECIPE,
            collection_ids=['fit0','fit1'],pairs=[dict(question_id=q,neutral={'style':'neutral'},emotion={'style':'angry'})
                for q in ('fit0','fit1')],averaging='matched')
        (source/'run_config.json').write_text(json.dumps(plan))
        torch.save(dict(completed_ids=plan['collection_ids'],difference_sum=vector['pair_mean_sums']['angry'].double(),
            rows=[dict(question_id=q,neutral_response='#### 2',emotion_response='#### 2') for q in plan['collection_ids']]),source/'collection-stream.pt')
        return source,bundle

    def test_collect_resume_control_and_exact_heldout_baselines(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            source,bundle=self.fixture(root)
            output=root/'new-pooling'
            parent,plans,job,baseline=active.prepare(source,bundle,output,[31])
            old_configs=(source/'run_config.json').read_bytes()
            original=torch.load(source/'style-vectors.pt',weights_only=True)
            delta=original['pair_mean_sums']['angry']/2
            def generate(record,*args):
                return record['style'],{}
            def capture(model,generate,decoder,budget):
                style,_=generate()
                h=torch.ones(32,1,4,3)+(delta.unsqueeze(1) if style=='angry' else 0)
                current=torch.ones(32,1,3)*(9 if style=='angry' else 2)
                return '#### 2',{},h,current
            decoder=SimpleNamespace()
            modules={'src':SimpleNamespace(fast_dllm=SimpleNamespace(diffa_generate=decoder)),
                     'src.fast_dllm':SimpleNamespace(diffa_generate=decoder),'src.fast_dllm.diffa_generate':decoder}
            with patch.dict('sys.modules',modules), \
                 patch.object(active.vanilla.task,'generate',side_effect=generate), \
                 patch.object(active,'capture_poolings',side_effect=capture) as capture_call:
                active.collect(parent,plans,source,output,None,None,None)
                active.collect(parent,plans,source,output,None,None,None)
            self.assertEqual(capture_call.call_count,4)
            full,_=active.winners.read_vector(output/'full-buffer/style-vectors.pt','angry',31,'answer')
            current,_=active.winners.read_vector(output/'active-block/style-vectors.pt','angry',31,'answer')
            torch.testing.assert_close(full,torch.full((3,),5.))
            torch.testing.assert_close(current,torch.full((3,),7.))
            self.assertEqual((source/'run_config.json').read_bytes(),old_configs)
            jobs=active.steering_jobs(plans,job,baseline,output,[31])
            self.assertEqual(len(jobs),2)
            for folder,selected in jobs:
                self.assertEqual([r['question_id'] for r in selected['target']['records']],['q0','q1'])
                self.assertEqual(active.read_rows(folder/'results.jsonl'),baseline)
                before=(folder/'results.jsonl').read_bytes()
                active.steering_jobs(plans,job,baseline,output,[31],create=False)
                self.assertEqual((folder/'results.jsonl').read_bytes(),before)
            self.assertEqual([j['layer'] for _,j in jobs],[31,31])
            with self.assertRaisesRegex(ValueError,'Frozen settings'):
                active.prepare(source,bundle,output,[28])
