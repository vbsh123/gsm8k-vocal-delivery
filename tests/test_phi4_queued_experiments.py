from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_ifeval_all_layers as layers
import phi4_gsm_combo as combo
import phi4_queued_workers as queue
import phi4_task_steering as task
from kimi_bbh_layer_diagnostic import digest


class QueuedPhiTests(unittest.TestCase):
    def test_three_workers_have_disjoint_collection_and_all64_settings(self):
        for items,count in ((range(254),254),(layers.jobs(),64)):
            parts=[layers.assigned(items,i) for i in range(3)]
            combined=[x for part in parts for x in part]
            self.assertEqual(len(combined),count);self.assertEqual(len(set(combined)),count)
            self.assertEqual(set(combined),set(items))
        self.assertEqual([len(layers.assigned(layers.jobs(),i)) for i in range(3)],[22,21,21])
        self.assertEqual(set(layers.jobs()),{(i,g) for i in range(32) for g in ('audio','answer')})

    def test_shared_capture_merge_and_layer_slice_avoid_recollection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=layers.cache_root(root);(cache/'pairs').mkdir(parents=True)
            pairs=[dict(question_id=q,voice='Jenny',neutral=dict(audio_sha256='n'),angry=dict(audio_sha256='a')) for q in ('a','b')]
            data=dict(collection=pairs,prompt='original',max_new_tokens=1024)
            config=layers.capture_config(data);(cache/'run_config.json').write_text(json.dumps(config));sha=digest(cache/'run_config.json')
            for i,pair in enumerate(pairs):
                neutral=torch.zeros(task.native_hooks.SHAPE);angry=torch.full_like(neutral,2.+i*4)
                angry[:,1]+=10;angry[30]+=5
                torch.save(dict(run_config_sha256=sha,question_id=pair['question_id'],voice='Jenny',audio_sha256=dict(neutral='n',angry='a'),
                    traces=dict(neutral=dict(means=neutral,hook_forwards=1),angry=dict(means=angry,hook_forwards=1))),cache/'pairs'/f'{i:04d}.pt')
            with redirect_stdout(io.StringIO()):layers.merge(root)
            plan=dict(config,protocol='evaluation',layer=30,group='audio',vector_definition='original')
            output=root/'evaluation';output.mkdir();(output/'run_config.json').write_text(json.dumps(plan))
            with patch.object(task.engine,'collect',side_effect=AssertionError('must reuse all-layer vector')):
                vector=task.collection_vector(plan,output,None,root/'collection-cache')
            torch.testing.assert_close(vector,torch.full((3072,),19.))
            state=torch.load(output/'style-vectors.pt',weights_only=True)
            self.assertEqual(state['group'],'audio');self.assertEqual(state['collection_question_ids'],['a','b'])
            changed=dict(plan,max_new_tokens=512);other=root/'other';other.mkdir();(other/'run_config.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):task.collection_vector(changed,other,None,root/'collection-cache')

    def test_bbh_combined_only_reuses_neutral_and_scores_choice_task(self):
        for name,letters in (('bbh3','ABC'),('bbh5','ABCDE')):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                root=Path(d);sources=[]
                collection=[dict(question_id=f'train{i}') for i in range(125)]
                records=[dict(question_id=f'test{i}',voice='Jenny',audio_sha256=f'a{i}',audio_file='unused.wav',reference_answer='A',condition='neutral',style='neutral') for i in range(125)]
                bank={r['question_id']:dict(original_input='Which?\nOptions:\n'+'\n'.join(f'({x}) Choice {x}' for x in letters),original_target='(A)') for r in records}
                recipe=dict(model='phi',revision='rev')
                for layer,group in ((0,'audio'),(30,'answer')):
                    folder=root/group;folder.mkdir();sources.append(folder)
                    config=dict(collection=collection,recipe=recipe,prompt='prompt',seed=42,max_new_tokens=512,recordings=records,bank=bank)
                    (folder/'run_config.json').write_text(json.dumps(config));sha=digest(folder/'run_config.json')
                    torch.save(dict(run_config_sha256=sha,collection_question_ids=[r['question_id'] for r in collection],layer=layer,group=group,
                        **recipe,angry_minus_neutral=torch.ones(3072)),folder/'style-vectors.pt')
                    (folder/'results.jsonl').write_text(''.join(json.dumps(dict(r,run_config_sha256=sha,prompt='prompt',seed=42,max_new_tokens=512,
                        **recipe,interventions=[],strength=0.,response='The answer is (A).',response_tokens=6))+'\n' for r in records))
                output=root/'combined';plan,vectors=combo.prepare(output,*sources,dataset=name,combined_only=True)
                self.assertEqual(set(plan['conditions']),{'neutral','combined'})
                observed=[]
                def generate(native,row,prompt,**kwargs):
                    observed.append(kwargs['interventions'])
                    return dict(response='The answer is (A).',response_tokens=6)
                with patch.object(combo.hooks,'generate',side_effect=generate),redirect_stdout(io.StringIO()):
                    combo.run(plan,output,vectors,native=object());combo.run(plan,output,vectors,native=object())
                    combo.report(plan,output)
                self.assertEqual(len(observed),125)
                self.assertEqual([(r['layer'],r['group'],r['strength']) for r in observed[0]],[(0,'audio',-1.),(30,'answer',-2.)])
                summary=json.loads((output/'summary.json').read_text())
                self.assertEqual(summary['n'],125)
                self.assertEqual({x['metric'] for x in summary['paired']},{'flexible-extract','strict-match','format-compliant'})
                self.assertIn('logical_deduction_',summary['scorer']['task'])

    def test_gpu_admission_waits_until_existing_processes_finish(self):
        with tempfile.TemporaryDirectory() as d:
            previous=Path.cwd();os.chdir(d)
            try:
                with patch.object(queue.subprocess,'check_output',side_effect=['123\n','']),patch.object(queue.time,'sleep') as sleep,redirect_stdout(io.StringIO()):
                    with queue.gpu_queue():self.assertTrue((Path('outputs')/'jenny-task-vectors.lock').exists())
                sleep.assert_called_once_with(30)
            finally:os.chdir(previous)

    def test_three_real_child_processes_wait_for_merge_release(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);release=root/'release';commands=[];ready=[]
            for i in range(3):
                marker=root/f'{i}.ready';done=root/f'{i}.done';ready.append(marker)
                program="import os,time;from pathlib import Path; p=Path("+repr(str(marker))+");p.write_text(str(os.getpid()));r=Path("+repr(str(release))+");deadline=time.monotonic()+8\nwhile not r.exists() and time.monotonic()<deadline:time.sleep(.02)\nassert r.exists();Path("+repr(str(done))+").write_text('done')"
                commands.append([sys.executable,'-c',program])
            checks=[]
            def merge():
                self.assertTrue(all(x.exists() for x in ready));self.assertFalse(any(root.glob('*.done')));checks.append(True)
            with redirect_stdout(io.StringIO()):queue.run_workers(commands,root,ready_paths=ready,release=release,on_ready=merge)
            self.assertEqual(checks,[True]);self.assertEqual(len(list(root.glob('*.done'))),3)
            self.assertEqual(len({x.read_text() for x in ready}),3)

    def test_failed_worker_terminates_the_other_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pidfile=root/'slow.pid'
            slow="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(10)"
            commands=[[sys.executable,'-c','import time;time.sleep(.2);raise SystemExit(7)'],[sys.executable,'-c',slow]]
            with self.assertRaisesRegex(RuntimeError,'7'),redirect_stdout(io.StringIO()):queue.run_workers(commands,root)
            if pidfile.exists():
                with self.assertRaises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)

if __name__=='__main__':unittest.main()
