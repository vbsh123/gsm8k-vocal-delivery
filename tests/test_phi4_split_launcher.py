import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SplitLauncherTests(unittest.TestCase):
    def launch(self, phase, fail=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'run_phi4_all_layers_split.sh'
            script.write_text((Path(__file__).resolve().parents[1]/script.name).read_text())
            log=root/'calls.jsonl'
            fake='''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(os.environ['PHI_SPLIT_TEST_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if os.environ.get('PHI_SPLIT_TEST_FAIL') and '--mode' in args and args[args.index('--mode')+1]=='collect-shard' and args[args.index('--worker-id')+1]=='1':raise SystemExit(7)
'''
            for name in ('.venv-diffa','.venv-bbh-phi4'):
                path=root/name/'bin/python';path.parent.mkdir(parents=True);path.write_text(fake);path.chmod(0o755)
            result=subprocess.run(['bash',str(script),phase],cwd=root,env=dict(os.environ,
                PHI_SPLIT_TEST_LOG=str(log),PHI_SPLIT_TEST_FAIL='1' if fail else ''),capture_output=True,text=True)
            calls=[json.loads(line) for line in log.read_text().splitlines()]
            modes=[args[args.index('--mode')+1] for args in calls]
            return result,calls,modes

    def test_collection_only_two_workers_then_merge_no_test(self):
        result,calls,modes=self.launch('collect')
        self.assertEqual(result.returncode,0,result.stderr)
        workers=[a[a.index('--worker-id')+1] for a in calls if '--worker-id' in a]
        self.assertEqual(sorted(workers),['0','1'])
        self.assertEqual(modes.count('collect-shard'),2)
        self.assertEqual(modes[-2:],['merge-vectors','summary'])
        self.assertNotIn('select',modes);self.assertNotIn('steer',modes);self.assertNotIn('references-shard',modes)

    def test_test_phase_references_then_select_then_two_steering_workers(self):
        result,calls,modes=self.launch('test')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(modes.count('references-shard'),2);self.assertEqual(modes.count('steer'),2)
        selected=modes.index('select')
        self.assertTrue(all(i<selected for i,m in enumerate(modes) if m=='references-shard'))
        self.assertTrue(all(i>selected for i,m in enumerate(modes) if m=='steer'))
        self.assertEqual(modes[-1],'summary')

    def test_failed_worker_prevents_incomplete_vector_merge(self):
        result,calls,modes=self.launch('collect',fail=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(modes.count('collect-shard'),2)
        self.assertNotIn('merge-vectors',modes)


if __name__=='__main__':unittest.main()
