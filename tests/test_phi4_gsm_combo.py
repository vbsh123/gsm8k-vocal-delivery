import unittest
from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace
import torch
import phi4_all_layer_diagnostic as hooks
import phi4_gsm_combo as run


class ComboTests(unittest.TestCase):
    def test_combined_hooks_audio_prefill_answer_cached_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self,h):return (h+1,'cache')
        layers=torch.nn.ModuleList([Block(),Block()]);mask=torch.tensor([True,True,False])
        with ExitStack() as stack:
            for layer,group,strength in ((0,'audio',-1.),(1,'answer',-2.)):
                stack.enter_context(hooks.hooks(layers,mask,target=layer,group=group,direction=torch.ones(3),strength=strength))
            h=torch.zeros(1,3,3)
            for layer in layers:
                h,cache=layer(h);self.assertEqual(cache,'cache')
            torch.testing.assert_close(h,torch.tensor([[[1.,1.,1.],[1.,1.,1.],[0.,0.,0.]]]))
            h=torch.zeros(1,1,3)
            for layer in layers:h=layer(h)[0]
            torch.testing.assert_close(h,torch.zeros_like(h))
        self.assertTrue(all(not layer._forward_hooks for layer in layers))

    def test_generate_installs_both_hooks_in_one_native_generation(self):
        class Batch(dict):
            def to(self,*a,**k):return self
        class Processor:
            def __call__(self,**kwargs):return Batch(input_ids=torch.tensor([[200011,7]]),audio_embed_sizes=torch.tensor([1]))
            def batch_decode(self,*args,**kwargs):return ['#### 2']
        class Block(torch.nn.Module):
            def forward(self,h):return h
        layers=torch.nn.ModuleList([Block(),Block()]);observed=[]
        def generate(**kwargs):
            h=torch.zeros(1,2,3)
            for layer in layers:h=layer(h)
            observed.append(h.clone());return torch.tensor([[200011,7,8]])
        native=SimpleNamespace(processor=Processor(),layers=layers,config=None,model=SimpleNamespace(device='cpu',generate=generate))
        changes=[dict(layer=0,group='audio',strength=-1.,direction=torch.ones(3)),dict(layer=1,group='answer',strength=-2.,direction=torch.ones(3))]
        with patch.object(hooks,'digest',return_value='hash'),patch.object(hooks.engine,'seed'),patch.object(hooks.engine,'load_audio',return_value=[]):
            trace=hooks.generate(native,dict(audio_file='file',audio_sha256='hash'),'prompt',interventions=changes)
        self.assertEqual(len(observed),1)
        torch.testing.assert_close(observed[0],torch.tensor([[[-1.,-1.,-1.],[-2.,-2.,-2.]]]))
        self.assertEqual(trace['response_tokens'],1)
        self.assertTrue(all(not x._forward_hooks for x in layers))

    def test_control_reuse_rejects_prompt_audio_or_intervention_changes(self):
        recipe=dict(model='phi',revision='rev');plan=dict(prompt='prompt',recipe=recipe)
        original=dict(audio_sha256='audio',reference_answer='2')
        row=dict(original,run_config_sha256='config',prompt='prompt',seed=42,max_new_tokens=512,**recipe,interventions=[])
        self.assertTrue(run.compatible(row,'config',plan,original,[]))
        for update in (dict(prompt='other'),dict(audio_sha256='other'),dict(interventions=[dict(layer=0)]),dict(max_new_tokens=1024)):
            self.assertFalse(run.compatible(dict(row,**update),'config',plan,original,[]))

if __name__=='__main__':unittest.main()
