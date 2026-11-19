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
