import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

import collect_step_vectors as run


class StepVectorTests(unittest.TestCase):
    def test_collection_is_deterministic_disjoint_and_does_not_filter_by_outcome(self):
        ids=[f'q{i}' for i in range(500)];test=[f'q{i}' for i in range(500,900)]
        chosen=run.select_ids(ids,test,400)
        self.assertEqual(len(set(chosen)),400)
        self.assertEqual(chosen,run.select_ids(ids,test,400))
        self.assertFalse(set(chosen)&set(test))
        with self.assertRaises(ValueError):run.select_ids(ids,chosen,400)

    def test_hooks_are_passive_audio_prefill_only_and_last_state_once_per_forward(self):
        class Block(torch.nn.Module):
            def forward(self,h):return (h+1,)
        class Native(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block(),Block()])
            def forward(self,input_ids):
                h=torch.arange(input_ids.shape[1],dtype=torch.float32).reshape(1,-1,1)
                for layer in self.model.layers:h=layer(h)[0]
                return h
        model=Native();first=torch.tensor([[1,151688,151690,151690,151689,4]])
        second=torch.cat([first,torch.tensor([[9]])],dim=1)
        baseline=model(first)
        with run.observe(model) as states:
            actual=model(first);model(second)
        self.assertTrue(torch.equal(actual,baseline))
        self.assertEqual(states['counts'],{0:2,1:2})
        self.assertEqual(float(states['initial'][0][0,0]),6.)
        self.assertEqual(float(states['initial'][0][1,0]),3.5)
        self.assertEqual(float(states['totals'][0][0]/2),6.5)
        self.assertFalse(model._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        with self.assertRaises(RuntimeError):
            with run.observe(model):raise RuntimeError('failure')
        self.assertFalse(model._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_fit_uses_equal_question_weight_and_preserves_direction_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pairs').mkdir()
            pairs=[dict(question_id=q,neutral=dict(question_id=q,style='neutral',answer='5'),angry=dict(question_id=q,style='angry',answer='5')) for q in ('a','b')]
            plan=dict(pairs=pairs,recipe=dict(model='step',revision='fixed'),collection_ids=['a','b'],evaluation_ids=['heldout'],averaging='independent full generation means')
            run.freeze(root/'run_config.json',plan);h=run.digest(root/'run_config.json')
            for i,delta in enumerate((2.,6.)):
                traces={}
                for style,offset,count in [('neutral',0.,100),('angry',delta,2)]:
                    traces[style]=dict(initial=torch.ones(2,2,1)+offset,mean=torch.ones(2,2,1)+offset,
                        response='#### 5',response_tokens=5,text_tokens=4,generated_token_ids=[1,2],ended_with_eos=True,forwards=count)
                torch.save(dict(question_id=pairs[i]['question_id'],run_config_sha256=h,traces=traces),root/'pairs'/f'{i:05d}.pt')
            run.fit(plan,root)
            v=torch.load(root/'style-vectors.pt',weights_only=True)
            self.assertTrue(torch.allclose(v['angry_minus_neutral'],torch.full((2,2,1),4.)))
            self.assertEqual(len(run.read_rows(root/'results.jsonl')),4)
            before=run.digest(root/'style-vectors.pt');run.fit(plan,root)
            self.assertEqual(run.digest(root/'style-vectors.pt'),before)
            self.assertTrue((root/'layer-drift.csv').is_file())

    def test_collection_waits_for_current_step_and_gpu_before_loading(self):
        import step_audio_adapter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();evaluation=root/'current';evaluation.mkdir()
            run.freeze(out/'run_config.json',dict(pairs=[]));events=[]
            args=SimpleNamespace(output=out,evaluation=evaluation)
            plan=dict(pairs=[dict(question_id='q')])
            def lock(handle,operation):
                events.append('current' if Path(handle.name).parent==evaluation else 'shared')
            def load(path):
                events.append('load');raise RuntimeError('stop before actual model load')
            with patch.dict('sys.modules',{'transformers':SimpleNamespace(__version__=run.MODELS['step']['transformers'])}),patch.object(run.fcntl,'flock',side_effect=lock), \
                 patch.object(run.subprocess,'check_output',return_value=''),patch.object(run,'checkpoint',return_value='/cached'), \
                 patch.object(step_audio_adapter,'make_generator',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'stop before'):run.collect(plan,args)
            self.assertEqual(events,['current','shared','load'])


if __name__=='__main__':unittest.main()
