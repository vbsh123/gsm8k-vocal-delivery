"""Verify preview/accuracy/audit order and the GPU lock without running inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SpeedLauncherTests(unittest.TestCase):
    def test_preview_precedes_accuracy_and_cpu_audit_runs_after_lock_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copy2(Path(__file__).resolve().parents[1]/'run_speed_checks.sh',root)
            (root/'.venv-diffa/bin').mkdir(parents=True)
            (root/'fake-bin').mkdir()
            nvidia=root/'fake-bin/nvidia-smi'
            nvidia.write_text('#!/bin/sh\nexit 0\n')
            nvidia.chmod(0o755)
            python=root/'.venv-diffa/bin/python'
            python.write_text('''#!/usr/bin/env python3
import json,sys,subprocess
from pathlib import Path
args=sys.argv[1:]
with Path('calls.jsonl').open('a') as stream: stream.write(json.dumps(args)+'\\n')
script=next(a for a in args if a.endswith('.py') or a=='-')
if script=='prepare_spoken_gsm.py':
    path=Path('outputs/spoken-gsm-original/manifest.jsonl')
    path.parent.mkdir(parents=True)
    path.touch()
if script=='evaluate_audio_speed.py':
    assert subprocess.run(['flock','-n','outputs/jenny-task-vectors.lock','true']).returncode!=0
if script=='audit_speed_pitch.py':
    assert subprocess.run(['flock','-n','outputs/jenny-task-vectors.lock','true']).returncode==0
    folder=Path(args[args.index('--output')+1])
    folder.mkdir(parents=True)
    (folder/'index.html').write_text('fixture')
if script=='-': exec(sys.stdin.read())
''')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root/'fake-bin')+os.pathsep+os.environ['PATH'])
            subprocess.run(['bash',str(root/'run_speed_checks.sh')],check=True,env=env,stdout=subprocess.PIPE,text=True)
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            scripts=[next(a for a in args if a.endswith('.py') or a=='-') for args in calls]
            self.assertEqual(scripts,['prepare_spoken_gsm.py','speed_listen5.py','evaluate_audio_speed.py','audit_speed_pitch.py','-'])
            generation=calls[2]
            self.assertIn('--save-audio',generation)
            self.assertEqual(generation[generation.index('--num-examples')+1],'100')
            rates=generation[generation.index('--rates')+1:generation.index('--reuse-from')]
            self.assertEqual([float(r) for r in rates],[i/10 for i in range(5,21)])
            self.assertTrue((root/'outputs/spoken-gsm-speed100-extended-pitch-audit.zip').exists())


if __name__=='__main__':
    unittest.main()
