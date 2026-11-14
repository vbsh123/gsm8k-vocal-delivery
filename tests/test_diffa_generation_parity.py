"""Execute shared gen and zero-strength steering against the same instrumented model."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import random
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class DiffaGenerationParityTests(unittest.TestCase):
    def test_zero_steering_matches_eval_kwargs_features_rng_and_response(self):
        import numpy as np
        import torch
        from compare_cloud_accuracy import PROMPT
        from steer_audio import run_experiment
        calls=[];preprocessed=[]
        audio=np.arange(40,dtype=np.float32).reshape(20,2)/100
        class Batch:
            def __init__(self,a):
                self.input_features=torch.tensor(a).reshape(1,1,-1)
                self.attention_mask=torch.ones(1,len(a),dtype=torch.long)
            def to(self,device):return self
        def extractor(a,**kwargs):
            preprocessed.append((a.copy(),kwargs));return Batch(a)
        class Block(torch.nn.Module):
            def forward(self,h):return (h,None)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        src=ModuleType('src');src.__path__=[];fast=ModuleType('src.fast_dllm');fast.__path__=[]
        decoder=ModuleType('src.fast_dllm.diffa_generate');decoder.insert_audio_embedding=lambda t,s,l:t
        src.fast_dllm=fast;fast.diffa_generate=decoder
        class Model:
            config=SimpleNamespace(prompt_size=64)
            llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks)))
            def generate(self,tokenizer,**kwargs):
                calls.append(dict(kwargs,features=kwargs['input_audio_features'].clone(),mask=kwargs['input_audio_mask'].clone(),
                                  rng=(random.random(),float(np.random.random()),float(torch.rand(())))))
                decoder.insert_audio_embedding(torch.zeros(1,700,3),torch.zeros(1,128,3),torch.tensor([5]))
                for _ in range(3):
                    h=torch.ones(1,700,3)*18
                    for layer in blocks:h=layer(h)[0]
                return f'Final answer: {round(float(h[0,-1].mean()))}'
        model=Model()
        fake_librosa=SimpleNamespace(resample=lambda a,**kwargs:a*.5)
        fake_sf=SimpleNamespace(read=lambda *args,**kwargs:(audio.copy(),24000))
        modules={'librosa':fake_librosa,'soundfile':fake_sf,'src':src,'src.fast_dllm':fast,'src.fast_dllm.diffa_generate':decoder}
        original_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            if args and args[0]=='cuda':args=('cpu',*args[1:])
            return original_to(tensor,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()),patch.dict('sys.modules',modules):
            import extract_vectors
            root=Path(d);checkpoint=root/'vectors.pt'
            state=dict(prompt=PROMPT,max_new_tokens=512,groups=['answer'],pair_counts={'angry':torch.ones(32,dtype=torch.long)},
                       pair_mean_sums={'angry':torch.zeros(32,1,3)},extraction_question_ids=[])
            torch.save(state,checkpoint)
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            reference=extract_vectors.gen(audio.copy(),24000,model,None,extractor,PROMPT,512)
            recordings=[dict(clip_id='c',question_id='q',audio_path='test.wav',answer='18')]
            conditions=[dict(name='baseline',interventions=[]),dict(name='zero',interventions=[dict(layer=31,group='answer',strength=0)])]
            with patch.object(extract_vectors,'load_model',return_value=(model,None,extractor)),patch.object(torch.Tensor,'to',cpu_to):
                run_experiment(recordings,conditions,checkpoint,root)
            from steer import read_rows
            rows=read_rows(root/'results.jsonl')
        self.assertEqual([r['response'] for r in rows],[reference,reference])
        self.assertEqual([r['seed'] for r in rows],[42,42])
        self.assertEqual([r['steered_forwards'] for r in rows],[0,0])
        self.assertEqual(len(calls),3)
        for call in calls:
            self.assertEqual({k:call[k] for k in ('system_prompt','input_texts','steps','block_length','max_new_tokens','temperature','accelerate')},
                dict(system_prompt='',input_texts=[PROMPT],steps=128,block_length=32,max_new_tokens=512,temperature=0.,accelerate='fast_dllm'))
            torch.testing.assert_close(call['features'],calls[0]['features'])
            torch.testing.assert_close(call['mask'],calls[0]['mask'])
            self.assertEqual(call['rng'],calls[0]['rng'])
        np.testing.assert_array_equal(preprocessed[0][0],preprocessed[1][0])
        self.assertEqual(preprocessed[0][1],preprocessed[1][1])
        self.assertFalse(any(layer._forward_hooks for layer in blocks))


if __name__=='__main__':unittest.main()
