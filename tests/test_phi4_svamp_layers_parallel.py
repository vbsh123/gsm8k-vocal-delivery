"""Verify disjoint work, simultaneous launch, capture reuse, and persistent models."""
from contextlib import redirect_stdout
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
import torch
import phi4_svamp_layers_parallel as nightly
import phi4_task_steering as task


class NightlyTests(unittest.TestCase):
    def test_disjoint_complete_partition(self):
        layers=[i for i in range(32) if i not in (17,28,29,30)]
        groups=nightly.partitions(layers,3)
        self.assertEqual(sorted(sum(groups,[])),layers)
        self.assertEqual(len(set(sum(groups,[]))),len(layers))
        self.assertEqual([len(g) for g in groups],[10,9,9])

    def test_launches_all_three_before_waiting(self):
        events=[]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=SimpleNamespace(output=Path(tmp),data_root=Path(tmp)/'data',layers=[0,1,2,3],workers=3,
                strengths=[-1.,1.],collection_size=300,test_size=300)
            def start(command,**kwargs):
                i=int(command[command.index('--worker-id')+1]);events.append(f'start{i}')
                process=MagicMock();process.pid=i;process.poll.return_value=0
                process.wait.side_effect=lambda:events.append(f'wait{i}') or 0
                return process
            with patch.object(nightly.subprocess,'run') as prepare,patch.object(nightly.subprocess,'Popen',side_effect=start):
                nightly.launch(args)
            prepare.assert_called_once()
            self.assertEqual(events[:6],['start0','start1','start2','wait0','wait1','wait2'])

    def test_collection_generates_once_per_delivery_for_multiple_layers_and_resumes(self):
        from kimi_bbh_layer_diagnostic import digest
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            pair=dict(question_id='train',voice='Jenny',neutral=dict(audio_sha256='n'),angry=dict(audio_sha256='a'))
            plans=[dict(task='svamp',collection=[pair],recipe=dict(model='phi',revision='fixed'),prompt='Prompt',seed=42,
                max_new_tokens=512,layer=i,group='answer',vector_definition='raw mean') for i in (2,6)]
            calls=[]
            def generate(native,row,prompt,**kwargs):
                calls.append(kwargs['capture_layers'])
                value=1. if row['audio_sha256']=='n' else 3.
                return dict(response='2',response_tokens=1,hook_forwards=1,means=torch.full((2,2,3072),value))
            with patch.object(task.native_hooks,'generate',side_effect=generate):
                nightly.collect_together(plans,root,SimpleNamespace(native=object()))
                nightly.collect_together(plans,root,SimpleNamespace(native=object()))
            self.assertEqual(calls,[[2,6],[2,6]])
            for layer in (2,6):
                folder=root/'svamp'/f'L{layer}-answer'
                saved=torch.load(folder/'style-vectors.pt',weights_only=True)
                self.assertTrue(torch.equal(saved['angry_minus_neutral'],torch.full((3072,),2.)))
                self.assertEqual(saved['run_config_sha256'],digest(folder/'run_config.json'))

    def test_evaluate_reuses_model_and_updates_target_plan(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            plan=dict(task='svamp',layer=28,group='answer',prompt='Prompt',max_new_tokens=512,recipe=dict(model='phi',revision='fixed'))
            native=SimpleNamespace(plan=None,generate=MagicMock(return_value=dict(response='2',response_tokens=1)))
            row=dict(strength=-1.,question_id='test',condition='steered')
            with patch.object(task,'reuse'),patch.object(task,'completed',return_value=({'test':row},{})),\
                 patch.object(task,'Native') as constructor,patch.object(task,'collection_vector',return_value=torch.zeros(3072)),\
                 patch.object(task.engine,'version',return_value='test'),patch.object(torch.cuda,'get_device_name',return_value='fake'),\
                 patch.object(task,'digest',return_value='hash'):
                task.evaluate(plan,root,root/'cache',native=native)
            constructor.assert_not_called()
            self.assertIs(native.plan,plan)
            native.generate.assert_called_once()
            self.assertIn('"response": "2"',(root/'results.jsonl').read_text())


if __name__=='__main__':unittest.main()
