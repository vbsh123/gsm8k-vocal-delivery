from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diagnose_minicpm_gsm100 as run


class MiniDiagnosticTests(unittest.TestCase):
    def model(self):
        class Block(torch.nn.Module):
            def forward(self,h):return h+1,None
        class LLM(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([Block(),Block()]))
            def forward(self,inputs_embeds,attention_mask=None):
                h=inputs_embeds
                for layer in self.model.layers:h=layer(h)[0]
                return SimpleNamespace(logits=h)
        return SimpleNamespace(llm=LLM(),get_omni_embedding=lambda data,input_embeddings,**kwargs:input_embeddings)

    def test_native_prefill_audio_bounds_cached_answer_means_and_cleanup(self):
        model=self.model();original=model.get_omni_embedding
        h=torch.arange(15.).reshape(1,5,3)
        with run.observe(model) as (first,sums,counts,meta):
            model.get_omni_embedding({'audio_bounds':[[[1,3]]]},input_embeddings=h)
            model.llm(inputs_embeds=h,attention_mask=torch.ones(1,5,dtype=torch.long))
            model.llm(inputs_embeds=torch.ones(1,1,3)*20,attention_mask=torch.ones(1,6,dtype=torch.long))
        self.assertEqual(meta['input_tokens'],5);self.assertEqual(meta['audio_tokens'],2)
        torch.testing.assert_close(first[0][0],h[0,-1]+1)
        torch.testing.assert_close(first[0][1],(h[0,1:3]+1).mean(0))
        torch.testing.assert_close(sums[0]/counts[0],((h[0,-1]+1)+21)/2)
        self.assertEqual(counts,{0:2,1:2})
        self.assertIs(model.get_omni_embedding,original)
        self.assertFalse(model.llm._forward_hooks);self.assertFalse(model.llm._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in model.llm.model.layers))

    def test_failed_native_forward_restores_hooks(self):
        model=self.model();original=model.get_omni_embedding
        with self.assertRaisesRegex(ValueError,'embedding forward'):
            with run.observe(model):model.llm(inputs_embeds=torch.ones(1,5,3))
        self.assertIs(model.get_omni_embedding,original);self.assertFalse(model.llm._forward_hooks)

    def test_no_gpu_wait_allocator_cap_and_headroom_failure(self):
        with patch.object(torch.cuda,'mem_get_info',return_value=(28*2**30,48*2**30)),patch.object(torch.cuda,'set_per_process_memory_fraction') as limit:
            run.cap_gpu(22);limit.assert_called_once_with(22/48,device=0)
        with patch.object(torch.cuda,'mem_get_info',return_value=(20*2**30,48*2**30)),patch.object(torch.cuda,'set_per_process_memory_fraction') as limit:
            with self.assertRaisesRegex(RuntimeError,'without waiting'):run.cap_gpu(22)
            limit.assert_not_called()

    def test_same100_frozen_prompts_initial_and_generation_directions(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);parent=root/'parent';parent.mkdir();output=root/'out';output.mkdir()
            pairs=[dict(question_id=f'q{i}',voice='Jenny',neutral=dict(audio_sha256=f'n{i}'),angry=dict(audio_sha256=f'a{i}')) for i in range(100)]
            source=dict(protocol=run.PHI_PROTOCOL,pairs=pairs,bank={p['question_id']:dict(question='Q',answer='2') for p in pairs},
                        reasoning_prompt='Reason with #### answer',answer_only_prompt='Answer only with #### answer')
            (parent/'run_config.json').write_text(json.dumps(source))
            args=SimpleNamespace(parent=parent,output=output,gpu_budget_gib=22)
            plan=run.prepare(args)
            self.assertEqual(plan['pairs'],pairs);self.assertEqual(plan['reasoning_prompt'],source['reasoning_prompt'])
            self.assertEqual(plan['native_options']['enable_thinking'],False)
            self.assertEqual(plan,run.prepare(args))
            (output/'states').mkdir()
            shape=(2,2,3);r=torch.tensor([3.,4.,0.]).expand(shape);e=torch.tensor([-4.,0.,0.]).expand(shape)
            for i,pair in enumerate(pairs):
                n=torch.full(shape,float(i+10))
                for c,h in zip(run.CONDITIONS,(n,n-r,n+e)):
                    t=dict(run.metadata(plan,pair,c),states=h,generation_answer_mean=2*h[:,0],
                           forwards=2,response='#### 2',response_tokens=4)
                    torch.save(t,run.state_path(output,i,c))
            with patch.object(run,'SHAPE',shape):run.summarize(plan,output)
            v=torch.load(output/'style-vectors.pt',weights_only=True)
            torch.testing.assert_close(v['reasoning_minus_answer_only'],r)
            torch.testing.assert_close(v['angry_minus_neutral'],e)
            torch.testing.assert_close(v['generation_angry_minus_neutral'],2*e[:,0])
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['pairs'],100);self.assertEqual(len(summary['layers']),4)
            self.assertAlmostEqual(summary['layers'][0]['cosine'],-.6)
            self.assertEqual(len((output/'results.jsonl').read_text().splitlines()),300)


if __name__=='__main__':unittest.main()
