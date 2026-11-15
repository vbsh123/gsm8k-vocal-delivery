from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_bbh_reasoning200 as run
from rescore_audio_steering import numeric_within_tolerance
from wait_for_experiments import active_jobs


class ReasoningTaskTests(unittest.TestCase):
    def test_seeded_selection_preserves_truth_problem_and_never_adds_gold_to_audio(self):
        example = dict(input='Alice tells the truth. Does Alice tell the truth?', target='Yes')
        data = json.dumps(dict(examples=[example]*250)).encode()
        with patch.dict(run.CHECKSUMS, web_of_lies=hashlib.sha256(data).hexdigest()):
            chosen = run.questions(data, 'web_of_lies')
            self.assertEqual(chosen, run.questions(data, 'web_of_lies'))
        self.assertEqual(len(chosen), 200)
        self.assertEqual(len({r['question_id'] for r in chosen}), 200)
        self.assertTrue(all(r['question'] == example['input'] for r in chosen))
        self.assertEqual(chosen[0]['answer'], 'Yes')

    def test_tracking_choice_text_and_constraints_preserved(self):
        example = dict(input='Alice and Bob swap books. Who has red?\nOptions:\n(A) Alice\n(B) Bob\n(C) Claire', target='(B)')
        data = json.dumps(dict(examples=[example]*250)).encode()
        task = run.TASKS[1]
        with patch.dict(run.CHECKSUMS, {task: hashlib.sha256(data).hexdigest()}):
            chosen = run.questions(data, task)
        self.assertEqual(chosen[0]['question'], 'Alice and Bob swap books. Who has red?\nOptions:\nOption A: Alice\nOption B: Bob\nOption C: Claire')
        self.assertEqual(chosen[0]['original_target'], '(B)')
        self.assertIn('The answer is (<letter>).', run.PROMPTS[task])

    def test_wait_detects_ifbench_worker_and_launcher_but_not_tail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'repo'; root.mkdir(); proc = Path(d)/'proc'; proc.mkdir()
            for pid, argv in [(801, ['python', 'run_spoken_ifbench.py']),
                              (802, ['bash', 'run_ifbench_after_ifeval.sh']),
                              (803, ['tail', '-f', 'ifbench.log'])]:
                entry = proc/str(pid); entry.mkdir(); (entry/'cwd').symlink_to(root)
                (entry/'stat').write_text(f'{pid} (worker) S 0 0')
                (entry/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv))
            self.assertEqual(active_jobs('ifbench', root, proc), [801, 802])

    def test_svamp_author_criterion_is_separate_and_handles_boundaries(self):
        for prediction, gold, expected in [('5', '5', True), ('5.1', '5', True), ('5.1001', '5', False),
            ('-1.1', '-1', True), ('$1,000.09', '1000', True), ('[invalid]', '5', False), ('0/0', '5', False)]:
            self.assertEqual(numeric_within_tolerance(prediction, gold), expected)

    def test_all_task_queue_uses_separate_subprocesses_in_fixed_order(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            with patch('sys.argv', ['test', '--output', d]), patch.object(run.subprocess, 'run') as child:
                run.main()
            self.assertEqual([c.args[0][-1] for c in child.call_args_list], list(run.TASKS))
            self.assertEqual(json.loads((Path(d)/'status.json').read_text())['status'], 'complete')

    def test_delivery_generation_keeps_same_prompt_seed_and_never_loads_vectors(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); audio = root/'clip.wav'; audio.write_bytes(b'audio')
            tasks = [dict(question_id='q', voice='Jenny', style=s, condition=s, answer='No',
                audio_file=str(audio), audio_sha256=run.digest(audio)) for s in run.STYLES]
            plan = dict(recordings=tasks, prompt=run.PROMPTS['web_of_lies'], bank=dict(q=dict(original_input='Truth?', original_target='No')))
            calls = []
            def generate(audio, sr, model, tokenizer, features, prompt, tokens, seconds):
                calls.append((prompt, tokens, seconds, torch.initial_seed(), random.random(), float(np.random.random())))
                return 'The answer is no.', dict(mode='native')
            class Scorer:
                def score(self, responses, docs):
                    return {'strict-match': [dict(correct=True, prediction='no')]*len(responses)}
            modules = {'soundfile': SimpleNamespace(read=lambda *a, **kw: (np.zeros(8), 16000)),
                       'extract_vectors': SimpleNamespace(load_model=lambda: (object(), None, None), gen_chunked=generate)}
            with patch.dict('sys.modules', modules), patch.object(run, 'scorer_for', return_value=Scorer()), \
                 patch.dict(run.os.environ, {'HF_HOME': str(root/'cache')}), patch.object(torch, 'load') as load:
                run.run(plan, root); run.run(plan, root)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c == calls[0] for c in calls))
            self.assertEqual(calls[0][:4], (plan['prompt'], 512, 30, 42))
            load.assert_not_called()
            self.assertEqual([r['style'] for r in run.read_rows(root/'results.jsonl')], list(run.STYLES))


@unittest.skipUnless(importlib.util.find_spec('lm_eval'), 'Needs installed harness')
class ActualTaskAPITests(unittest.TestCase):
    def test_both_new_tasks_accept_the_requested_formats(self):
        with tempfile.TemporaryDirectory() as d:
            truth = run.HarnessTaskScorer('bbh_cot_zeroshot_web_of_lies', Path(d)/'truth')
            values = truth.score(['The answer is yes.', 'The answer is no.'],
                [dict(input='Truth?', target=t) for t in ('Yes', 'No')])
            self.assertTrue(all(r['correct'] for pipeline in values.values() for r in pipeline))
            tracking = run.HarnessTaskScorer('bbh_cot_zeroshot_tracking_shuffled_objects_three_objects', Path(d)/'tracking')
            docs = [dict(input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green', target=f'({c})') for c in 'ABC']
            values = tracking.score([f'The answer is ({c}).' for c in 'ABC'], docs)
            self.assertTrue(all(r['correct'] for pipeline in values.values() for r in pipeline))


if __name__ == '__main__':
    unittest.main()
