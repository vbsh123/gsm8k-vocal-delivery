from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

from evaluate_commonsense_audio import PROMPT
from steer_commonsense_mean import split_recordings,main
from run_L31_timing import conditions


class CommonsenseMeanTests(unittest.TestCase):
    def records(self):
        return [dict(question_id=f'q{i:03}',voice='Jenny',style=s,answer='B',audio_path=f'{i}-{s}.wav')
                for i in range(500) for s in ('neutral','angry','happy','sad')]

    def test_fixed_split_is_disjoint_complete_and_independent_of_correctness(self):
        records=self.records();train,test=split_recordings(records,250)
        self.assertEqual((len(train),len(test)),(250,250))
        self.assertFalse({r['question_id'] for r in train}&{r['question_id'] for r in test})
        self.assertEqual(len({r['question_id'] for r in train}|{r['question_id'] for r in test}),500)
        self.assertTrue(all(r['style']=='neutral' for r in test))
        self.assertEqual(split_recordings(records,250),(train,test))

    def test_prepare_freezes_split_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);bench=root/'benchmark';bench.mkdir()
            (bench/'run_config.json').write_text(json.dumps(dict(prompt=PROMPT,seed=42,max_new_tokens=512,recordings=self.records())))
            with patch('sys.argv',['test','--benchmark',str(bench),'--output',str(root/'out'),'--prepare-only']):main()
            split=json.loads((root/'out/split.json').read_text())
            self.assertEqual((len(split['extraction']),len(split['test'])),(250,250))
            self.assertEqual(split['targets'],[dict(layer=12,group='acoustic',strength=1),dict(layer=31,group='answer',strength=1)])

    def test_strength_sweep_leaves_existing_timing_defaults_unchanged(self):
        self.assertEqual([r['schedule'] for r in conditions()[1:]],['early','middle','late'])
        rows=conditions(True)
        self.assertEqual([r['interventions'][0]['strength'] for r in rows[1:]],[.75,1.25,1.5])
        self.assertTrue(all('schedule' not in r for r in rows))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class CommonsenseRuntimeTests(unittest.TestCase):
    def test_collects_both_targets_and_evaluates_separately_with_reused_baseline(self):
        import torch
        import numpy as np
        from steer_commonsense_mean import run,TARGETS
        class Block(torch.nn.Module):
            def forward(self,h):return (h+.01,None)
        model=SimpleNamespace(config=SimpleNamespace(prompt_size=64),
            llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=torch.nn.ModuleList([Block() for _ in range(32)])))))
        src=ModuleType('src');src.__path__=[];fast=ModuleType('src.fast_dllm');fast.__path__=[]
        decoder=ModuleType('src.fast_dllm.diffa_generate');decoder.insert_audio_embedding=lambda text,speech,lengths:text
        src.fast_dllm=fast;fast.diffa_generate=decoder
        def generate(audio,*args):
            decoder.insert_audio_embedding(torch.zeros(1,700,4),torch.zeros(1,128,4),torch.tensor([5]))
            for _ in range(3 if audio[0]==2 else 2):
                h=torch.ones(1,700,4)*float(audio[0])
                for layer in model.llm_model.model.transformer.blocks:h=layer(h)[0]
            return 'Final answer: B'
        original_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            if args and args[0]=='cuda':args=('cpu',*args[1:])
            return original_to(tensor,*args,**kwargs)
        modules={'src':src,'src.fast_dllm':fast,'src.fast_dllm.diffa_generate':decoder,
                 'librosa':SimpleNamespace(),'soundfile':SimpleNamespace(read=lambda file,**kw:(np.array([2 if 'angry' in str(file) else 1],dtype=np.float32),16000)),
                 'extract_vectors':SimpleNamespace(load_model=lambda:(model,None,None),gen=generate)}
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'out';out.mkdir();bench=root/'benchmark';bench.mkdir()
            def recording(q,style):
                path=root/f'{q}-{style}.wav';path.write_bytes(b'waveform');stat=path.stat()
                return dict(question_id=q,style=style,voice='Jenny',answer='B',audio_file=str(path),audio_path=path.name,
                            audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns)
            train=dict(question_id='train',neutral=recording('train','neutral'),angry=recording('train','angry'))
            test=recording('test','neutral')
            baseline=dict(audio_path=test['audio_path'],question_id='test',style='neutral',voice='Jenny',
                          reference_answer='B',seed=42,prompt=PROMPT,response='Final answer: B',correct=True)
            (bench/'results.jsonl').write_text(json.dumps(baseline)+'\n')
            plan=dict(extraction=[train],test=[test],averaging='matched')
            with patch.dict('sys.modules',modules),patch.dict('os.environ',HF_HOME=str(root/'cache')),patch.object(torch.Tensor,'to',cpu_to):
                run(plan,bench,out)
            values=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
            self.assertEqual([r['condition'] for r in values],['baseline','L12-acoustic-a1','L31-answer-a1'])
            state=torch.load(out/'style-vectors.pt',weights_only=True)
            self.assertEqual(state['collected_targets'],TARGETS)
            self.assertEqual(int(state['pair_counts']['angry'][12]),1)
            self.assertEqual(int(state['pair_counts']['angry'][31]),1)
            self.assertEqual(state['extraction_question_ids'],['train'])
            pair=torch.load(out/'pairs/00000.pt',weights_only=True)
            self.assertEqual(pair['matched_forwards'],2)
            self.assertEqual(tuple(pair['difference'].shape),(2,4))


if __name__=='__main__':unittest.main()
