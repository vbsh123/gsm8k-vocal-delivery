import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phi4_opposite_recovery as run


class OppositeRecoveryTests(unittest.TestCase):
    def test_opposite_selection_and_both_inputs_exclude_original_cohort(self):
        rows=[]
        for q,n,a in [('opposite',True,False),('original',False,True),('both',True,True),('neither',False,False)]:
            for s,c in [('neutral',n),('angry',a)]:
                rows.append(dict(question_id=q,voice='Jenny',style=s,scores={'flexible-numeric':c}))
        cases=run.select_cases(rows)
        self.assertEqual(cases,[dict(question_id='opposite',voice='Jenny')])
        plan=dict(cases=cases,layers=list(range(32)),groups=['answer','audio'],input_styles=['neutral','angry'])
        original=dict(recordings=[dict(question_id='opposite',voice='Jenny',style=s) for s in plan['input_styles']])
        tasks=run.tasks_for(plan,original)
        self.assertEqual(len(tasks),128)
        self.assertEqual({key[2] for key in tasks},{'neutral','angry'})
        for key,row in tasks.items():self.assertEqual(key[2],row['style'])

    def test_focused_resume_skips_completed_shortlist_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);(output/'run_config.json').write_text('{}')
            plan=dict(layers=list(range(32)),groups=['answer','audio'],input_styles=['neutral','angry'])
            task=('q','Jenny','neutral',2,'answer');other=('q','Jenny','neutral',3,'audio')
            with patch.object(run,'completed',return_value=({task:{},other:{}},{task:{}})), \
                 patch.object(run.diagnostic.engine,'NativePhi') as model:
                run.run(plan,{},output,layers=[2],groups=['answer'])
                model.assert_not_called()
            self.assertEqual(json.loads((output/'execution-L2-answer.json').read_text())['layers'],[2])

    def test_stop_helper_matches_only_same_repo_wrapper_and_run(self):
        import stop_phi4_nightly_opposite as stop
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);proc=root/'proc';proc.mkdir();repo=root/'repo';repo.mkdir();other=root/'other';other.mkdir()
            records=[(101,repo,['bash','run_phi4_nightly_followups.sh']),
                (102,repo,['python','phi4_opposite_recovery.py','--mode','run']),
                (103,repo,['python','phi4_opposite_recovery.py','--mode','summary']),
                (104,other,['python','phi4_opposite_recovery.py','--mode','run']),
                (105,repo,['python','phi4_task_steering.py','--mode','run'])]
            for pid,cwd,args in records:
                path=proc/str(pid);path.mkdir();(path/'cwd').symlink_to(cwd)
                (path/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args)+b'\0')
            self.assertEqual(stop.targets(repo,proc),[(101,True),(102,False)])

    def test_prepare_reuses_existing_vector_freezes_receipts_and_is_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'original';source.mkdir();output=root/'opposite'
            original=dict(collection=[dict(question_id='train')],recordings=[],layers=list(range(32)),groups=['answer','audio'],
                prompt='Same prompt',recipe={'model':'Phi'})
            (source/'run_config.json').write_text(json.dumps(original));(source/'style-vectors.pt').write_bytes(b'original vector')
            rows=[dict(question_id='test',voice='Jenny',style=s,scores={'flexible-numeric':s=='neutral'}) for s in ('neutral','angry')]
            with patch.object(run.diagnostic,'references',return_value=({},{})), \
                 patch.object(run.diagnostic,'score_rows',return_value=(rows,{})):
                plan,_=run.prepare(source,output)
            self.assertEqual(plan['strength'],-1.)
            self.assertEqual(plan['cases'],[dict(question_id='test',voice='Jenny')])
            self.assertEqual(plan['vector_sha256'],run.digest(source/'style-vectors.pt'))
            self.assertFalse((output/'style-vectors.pt').exists())
            run.validate_source(plan)
            (source/'style-vectors.pt').write_bytes(b'changed vector')
            with self.assertRaisesRegex(ValueError,'evidence changed'):run.validate_source(plan)

class NightlyQueueTests(unittest.TestCase):
    def test_shared_gpu_lock_blocks_added_stages_then_order_is_sequential(self):
        import fcntl
        import os
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'run_phi4_nightly_followups.sh'
            script.write_text((Path(run.__file__).parent/script.name).read_text())
            (root/'outputs').mkdir();log=root/'calls'
            fake='''#!/usr/bin/env python3
import os,sys
with open(os.environ['NIGHTLY_TEST_LOG'],'a') as f:f.write(' '.join(sys.argv[1:])+'\\n')
'''
            for env in ('.venv-diffa','.venv-bbh-phi4'):
                path=root/env/'bin/python';path.parent.mkdir(parents=True);path.write_text(fake);path.chmod(0o755)
            for name in ('prepare_phi4_next_tasks.sh','run_phi4_prepared_tasks.sh'):
                (root/name).write_text('echo '+name+' "$@" >> "$NIGHTLY_TEST_LOG"\n')
            binary=root/'bin';binary.mkdir();smi=binary/'nvidia-smi';smi.write_text('#!/bin/sh\nexit 0\n');smi.chmod(0o755)
            with (root/'outputs/jenny-task-vectors.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                proc=subprocess.Popen(['bash',str(script)],cwd=root,env=dict(os.environ,NIGHTLY_TEST_LOG=str(log),PATH=str(binary)+':'+os.environ['PATH']),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    deadline=time.monotonic()+5
                    while (not log.exists() or len(log.read_text().splitlines())<2) and time.monotonic()<deadline:time.sleep(.02)
                    self.assertIsNone(proc.poll())
                    self.assertNotIn('phi4_opposite_recovery.py',log.read_text())
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    stdout,stderr=proc.communicate(timeout=10)
                    self.assertEqual(proc.returncode,0,stdout+stderr)
                finally:
                    if proc.poll() is None:proc.kill();proc.communicate()
            calls=log.read_text().splitlines()
            self.assertEqual(calls[0],'-u wait_for_experiments.py --kind phi-layer-nightly-inputs')
            self.assertEqual(calls[2:5],['-u phi4_opposite_recovery.py --mode '+mode for mode in ('prepare','run','summary')])
            self.assertEqual(calls[-2:],[ 'prepare_phi4_next_tasks.sh', 'run_phi4_prepared_tasks.sh --layer 17 --group answer --strength -1.5'])


if __name__=='__main__':unittest.main()
