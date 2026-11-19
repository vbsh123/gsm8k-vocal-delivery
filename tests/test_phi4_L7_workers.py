from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import run_phi4_gsm_L7 as run
from steer import read_rows


class L7WorkerTests(unittest.TestCase):
    def setup_output(self,root):
        output=root/'out';output.mkdir();tests=[dict(question_id=f'q{i}',voice='Jenny',style='neutral',audio_file='audio',audio_sha256=f'h{i}',reference_answer='2') for i in range(2)]
        plan=dict(protocol='test',test=tests,bank={r['question_id']:dict(question='Q',answer='2') for r in tests},
            recipe=dict(model='Phi',revision='fixed'),prompt='Prompt',seed=42,max_new_tokens=512,layer=7,group='answer',
            strengths={'neutral':0.,'neutral-L7-answer-a-1':-1.,'neutral-L7-answer-a-1.5':-1.5},collection=[],scoring='test')
        (output/'run_config.json').write_text(json.dumps(plan));(output/'style-vectors.pt').write_bytes(b'vector')
        def saved(row,c):return dict(row,response='#### 2',response_tokens=4,condition=c,protocol='test',
            run_config_sha256=run.digest(output/'run_config.json'),prompt='Prompt',seed=42,max_new_tokens=512,model='Phi',revision='fixed',strength=plan['strengths'][c],layer=7)
        rows=[saved(r,'neutral') for r in tests]+[saved(tests[0],'neutral-L7-answer-a-1')]
        (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return plan,output

    def test_workers_own_one_strength_reuse_serial_and_never_repair_other_worker(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            plan,out=self.setup_output(Path(d));calls=[];baseline=(out/'results.jsonl').read_bytes()
            def generate(row,prompt,direction,strength):
                calls.append((row['question_id'],strength));return dict(response='#### 2',response_tokens=4)
            native=SimpleNamespace(generate=generate)
            with patch.object(run,'fixed_vector',return_value=torch.ones(3072)),patch.object(run.engine,'NativePhi',return_value=native), \
                 patch.object(run.diagnostic.candidates,'concurrent_budget_now'),patch.object(torch.cuda,'get_device_name',return_value='fake'):
                run.execute(plan,out,0)
                self.assertEqual(calls,[('q1',-1.)])
                first=out/'results-worker0.jsonl'
                with first.open('ab') as stream:stream.write(b'{"partial":')
                before=first.read_bytes()
                # Worker1 repairs only its own interrupted tail.
                (out/'results-worker1.jsonl').write_text('{"partial":')
                run.execute(plan,out,1)
                self.assertEqual(calls,[('q1',-1.),('q0',-1.5),('q1',-1.5)])
                self.assertEqual(first.read_bytes(),before)
                run.execute(plan,out,1)
                self.assertEqual(len(calls),3)
            self.assertEqual((out/'results.jsonl').read_bytes(),baseline)
            self.assertEqual(len(run.completed(plan,out)[1]),6)
            for worker,strength in [(0,-1.),(1,-1.5)]:
                self.assertEqual({r['strength'] for r in read_rows(out/f'results-worker{worker}.jsonl')},{strength})

    def test_individual_summary_uses_pairs_without_waiting_for_other_strength(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            plan,out=self.setup_output(Path(d))
            def generate(row,prompt,direction,strength):return dict(response='#### 2',response_tokens=4)
            run.engine.evaluate(plan,out,SimpleNamespace(generate=generate),torch.ones(3072),run.digest(out/'run_config.json'),
                conditions=['neutral-L7-answer-a-1'],result_file=out/'results-worker0.jsonl',result_paths=run.result_files(out))
            class Scorer:
                provenance={}
                def __init__(self,*args):pass
                def score(self,responses,docs):return {m:[dict(correct=True,prediction='2') for _ in docs] for m in ('strict-match','flexible-extract')}
            with patch('rescore_audio_steering.HarnessTaskScorer',Scorer):
                run.engine.report(out,result_paths=run.result_files(out),conditions=['neutral','neutral-L7-answer-a-1'])
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['matched_questions'],2)
            self.assertEqual(len(summary['accuracy']),2)

    def test_launcher_starts_two_strength_workers_then_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'run_phi4_gsm_L7_split.sh';script.write_text((Path(run.__file__).parent/script.name).read_text())
            calls=root/'calls';fake='''#!/usr/bin/env python3
import os,sys,json
with open(os.environ['L7_TEST_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
'''
            for env in ('.venv-diffa','.venv-bbh-phi4'):
                path=root/env/'bin/python';path.parent.mkdir(parents=True);path.write_text(fake);path.chmod(0o755)
            bindir=root/'bin';bindir.mkdir();smi=bindir/'nvidia-smi';smi.write_text('#!/bin/sh\nexit 0\n');smi.chmod(0o755)
            result=subprocess.run(['bash',str(script)],cwd=root,env=dict(os.environ,L7_TEST_LOG=str(calls),PATH=str(bindir)+':'+os.environ['PATH']),capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(s) for s in calls.read_text().splitlines()]
            modes=[r[r.index('--mode')+1] for r in rows]
            self.assertEqual(modes,['prepare','worker','worker','summary'])
            self.assertEqual({r[r.index('--worker-id')+1] for r in rows if '--worker-id' in r},{'0','1'})


if __name__=='__main__':unittest.main()
