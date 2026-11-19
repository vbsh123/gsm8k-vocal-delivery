"""The sweep must preserve splits and share vectors/baselines without changing recipes."""
from contextlib import redirect_stdout
import io,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import phi4_sweep as sweep
import phi4_task_steering as task
from kimi_bbh_layer_diagnostic import freeze,digest


def fixture(root):
    def row(q,s):return dict(question_id=q,voice='Jenny',style=s,audio_file='audio',audio_sha256=q+s,reference_answer='2')
    pair=lambda q:dict(question_id=q,voice='Jenny',neutral=row(q,'neutral'),angry=row(q,'angry'))
    data=dict(task='multiarith',collection=[pair('train')],test=[dict(question_id='test',voice='Jenny',neutral=row('test','neutral'))],
        bank={'test':dict(question='Q',answer='2')},prompt='Prompt')
    path=root/'data_config.json';freeze(path,data);return data,path


class SweepTests(unittest.TestCase):
    def test_summary_wrapper_and_run_scoring_use_cpu_environment(self):
        import os,subprocess
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'run_phi4_sweep.sh'
            script.write_text((Path(sweep.__file__).parent/script.name).read_text())
            for env,label in [('.venv-diffa','cpu'),('.venv-bbh-phi4','phi')]:
                python=root/env/'bin/python';python.parent.mkdir(parents=True)
                python.write_text('#!/bin/sh\necho '+label+'\n');python.chmod(0o755)
            for flags,label in [(['--mode','summary'],'cpu'),(['--mode=summary'],'cpu'),(['--mode','prepare'],'cpu'),(['--mode','parallel'],'cpu'),([], 'phi')]:
                result=subprocess.check_output(['bash',str(script),*flags],text=True,env={k:v for k,v in os.environ.items() if k!='DIFFA_PYTHON'})
                self.assertEqual(result.strip(),label)
            args=SimpleNamespace(data_root=root/'data',output=root/'results')
            with patch.dict(os.environ,{},clear=True),patch.object(sweep.subprocess,'run') as run:
                sweep.score_in_cpu_environment(args,'gsm8k',6,-1.)
            command=run.call_args.args[0]
            self.assertEqual(command[0],'.venv-diffa/bin/python')
            self.assertIn('summary',command);self.assertIn('gsm8k',command)

    def test_parallel_prepares_once_then_starts_both_layers_before_waiting(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        events=[]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=SimpleNamespace(output=Path(tmp)/'out',data_root=Path(tmp)/'data',datasets=['gsm8k','ifeval'],layers=[17,30],strengths=[-1.,-1.5],no_wait=True)
            workers=[]
            def start(command,**kwargs):
                layer=command[command.index('--layers')+1];events.append('start'+layer)
                process=MagicMock();process.pid=int(layer);process.wait.side_effect=lambda: events.append('wait'+layer) or 0
                workers.append(command);return process
            with patch.object(sweep.subprocess,'run') as prepare,patch.object(sweep.subprocess,'Popen',side_effect=start):
                sweep.parallel(args)
            prepare.assert_called_once()
            self.assertEqual(events,['start17','start30','wait17','wait30'])
            for command in workers:
                self.assertIn('--no-wait',command);self.assertIn('ifeval',command);self.assertIn('-1.5',command)
            self.assertTrue((args.output/'worker-L17.log').exists());self.assertTrue((args.output/'worker-L30.log').exists())

    def test_different_jobs_can_run_but_duplicate_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with sweep.job_lock(root/'gsm8k/L6'):
                with sweep.job_lock(root/'multiarith/L7'):pass
                with self.assertRaisesRegex(SystemExit,'Already running'):
                    with sweep.job_lock(root/'gsm8k/L6'):pass
            with sweep.job_lock(root/'gsm8k/L6'):pass

    def test_no_wait_does_not_acquire_global_output_queue(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(sys,'argv',['phi4_sweep.py','--no-wait','--output',str(root)]),patch.object(sweep,'run') as run:
                sweep.main()
            run.assert_called_once()
            self.assertFalse((root/'queue.lock').exists())

    def test_collection_once_for_two_strengths_and_layer_isolation(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);data,path=fixture(root);calls=[]
            class Native:
                def generate(self,row,prompt,observe=False):
                    calls.append(row['style'])
                    return dict(response='#### 2',response_tokens=4,hook_forwards=2,mean=torch.full((3072,),2. if row['style']=='angry' else 0.))
            for layer,strength in [(2,-1.),(2,-1.5),(7,-1.)]:
                plan=task.make_plan(data,path,layer,'answer',strength);out=root/f'L{layer}a{strength}';out.mkdir();freeze(out/'run_config.json',plan)
                vector=task.collection_vector(plan,out,Native(),root/'cache')
                torch.testing.assert_close(vector,torch.full((3072,),2.))
                self.assertEqual(len(plan['recordings']),2)
            self.assertEqual(calls,['neutral','angry','neutral','angry'])

    def test_reuse_requires_matching_audio_prompt_and_unsteered_generation(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);data,path=fixture(root);plan=task.make_plan(data,path,2,'answer',-1.)
            out=root/'new';out.mkdir();freeze(out/'run_config.json',plan)
            original=dict(plan['recordings'][0],response='#### 2',response_tokens=4,prompt='Prompt',seed=42,max_new_tokens=512,
                model=plan['recipe']['model'],revision=plan['recipe']['revision'],interventions=[])
            source=root/'source.jsonl';source.write_text(json.dumps(dict(original,audio_sha256='wrong'))+'\n'+json.dumps(original)+'\n')
            with patch.object(sweep,'baseline_sources',return_value=[source]):
                sweep.reuse_baselines(plan,out,root);sweep.reuse_baselines(plan,out,root)
            saved=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(saved),1);self.assertEqual(saved[0]['audio_sha256'],'testneutral')
            self.assertEqual(saved[0]['reused_from'],str(source))


if __name__=='__main__':unittest.main()
