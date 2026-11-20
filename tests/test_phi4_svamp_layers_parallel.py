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

    def test_gpu_wait_happens_before_any_worker_starts(self):
        events=[]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=SimpleNamespace(output=Path(tmp),data_root=Path(tmp)/'data',layers=[0,1,2],workers=3,
                strengths=[-1.],collection_size=300,test_size=300,wait_for_gpu=True)
            def start(*a,**k):
                events.append('start');process=MagicMock();process.poll.return_value=0;process.wait.return_value=0
                return process
            with patch.object(nightly.subprocess,'run'),patch.object(nightly,'wait_for_gpu',side_effect=lambda stack:events.append('gpu-ready')),\
                 patch.object(nightly.subprocess,'Popen',side_effect=start):
                nightly.launch(args)
            self.assertEqual(events,['gpu-ready','start','start','start'])

    def test_gpu_wait_checks_active_processes_until_idle(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            with patch.object(nightly.Path,'open',return_value=(Path(tmp)/'gpu.lock').open('a')),\
                 patch.object(nightly.subprocess,'check_output',side_effect=['123\n','']),\
                 patch.object(nightly.time,'sleep') as sleep,ExitStack() as stack:
                nightly.wait_for_gpu(stack)
            sleep.assert_called_once_with(30)

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

    def test_audio_worker_preserves_group_in_collection_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'data/svamp/data_config.json';source.parent.mkdir(parents=True);source.write_text('{}')
            args=SimpleNamespace(output=root/'out',data_root=root/'data',layers=[1,2],workers=1,worker_id=0,group='audio',strengths=[-1.])
            with patch.object(nightly.sweep,'subset_data',return_value=({},source)),\
                 patch.object(nightly.task,'make_plan',return_value={'group':'audio'}) as plan,\
                 patch('evaluate_bbh_candidates.concurrent_budget_now'),patch.object(nightly.task,'Native') as model,\
                 patch.object(nightly,'collect_together') as collect,patch.object(nightly.sweep,'run') as evaluate:
                nightly.worker(args)
            self.assertTrue(all(call.args[3]=='audio' for call in plan.call_args_list))
            self.assertEqual(args.group,'audio')
            collect.assert_called_once();model.assert_called_once();evaluate.assert_called_once_with(args,native=model.return_value)

    def test_audio_parent_passes_group_to_every_worker(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=SimpleNamespace(output=Path(tmp),data_root=Path(tmp)/'data',layers=[1,2,3],workers=3,
                strengths=[-1.],collection_size=300,test_size=300,group='audio')
            process=MagicMock();process.poll.return_value=0;process.wait.return_value=0
            with patch.object(nightly.subprocess,'run') as prepare,patch.object(nightly.subprocess,'Popen',return_value=process) as launch:
                nightly.launch(args)
            for command in [prepare.call_args.args[0],*[call.args[0] for call in launch.call_args_list]]:
                self.assertEqual(command[command.index('--group')+1],'audio')

    def test_audio_collection_selects_audio_component_not_answer(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);pair=dict(question_id='q',voice='Jenny',neutral={'audio_sha256':'n'},angry={'audio_sha256':'a'})
            plan=dict(task='svamp',collection=[pair],recipe=dict(model='phi',revision='fixed'),prompt='Prompt',seed=42,
                max_new_tokens=512,layer=1,group='audio',vector_definition='raw mean')
            def generate(native,row,prompt,**kwargs):
                value=1. if row['audio_sha256']=='n' else 3.
                means=torch.stack((torch.full((3072,),value),torch.full((3072,),-2*value))).unsqueeze(0)
                return dict(response='2',response_tokens=1,hook_forwards=1,means=means)
            with patch.object(task.native_hooks,'generate',side_effect=generate):
                nightly.collect_together([plan],root,SimpleNamespace(native=object()))
            saved=torch.load(root/'svamp/L1-audio/style-vectors.pt',weights_only=True)
            self.assertTrue(torch.equal(saved['angry_minus_neutral'],torch.full((3072,),-4.)))

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
