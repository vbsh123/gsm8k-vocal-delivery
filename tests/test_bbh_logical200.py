from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import wave

import run_bbh_logical200 as run
from wait_for_experiments import active_jobs


class BBHTests(unittest.TestCase):
    def example(self):
        return dict(input='There are three objects. Red is left of blue.\nOptions:\n'
                    '(A) Red is leftmost\n(B) Blue is leftmost\n(C) Green is leftmost', target='(A)')

    def test_selection_preserves_constraints_and_options_without_spoken_gold(self):
        data = json.dumps(dict(examples=[self.example() for _ in range(250)]))
        selected = run.select_questions(data)
        self.assertEqual(len(selected), 200)
        self.assertEqual(len({r['question_id'] for r in selected}), 200)
        self.assertEqual(selected, run.select_questions(data))
        self.assertEqual(selected[0]['question'], 'There are three objects. Red is left of blue.\nOptions:\n'
            'Option A: Red is leftmost\nOption B: Blue is leftmost\nOption C: Green is leftmost')
        self.assertEqual(selected[0]['original_target'], '(A)')
        self.assertEqual(selected[0]['answer'], 'A')

    def test_correctness_is_separate_from_output_format(self):
        q = run.select_questions(json.dumps(dict(examples=[self.example()])), 1)[0]
        bare = run.score('(A)', q)
        formatted = run.score('Final answer: A', q)
        self.assertTrue(bare['answer_correct']); self.assertFalse(bare['correct'])
        self.assertTrue(formatted['answer_correct']); self.assertTrue(formatted['correct'])
        self.assertFalse(run.score('Final answer: B', q)['answer_correct'])
        self.assertTrue(run.score('Final answer: E', q)['answer_parse_failed'])

    def test_all_four_deliveries_required_to_fit_no_cropping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = []; bank = {}
            for qid in ('short', 'long'):
                bank[qid] = dict(question='Logic?', answer='A')
                for style in run.STYLES:
                    name = f'{qid}-{style}.wav'
                    with wave.open(str(root/name), 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                        audio.writeframes(b'\0\0' * (3001 if qid == 'long' and style == 'sad' else 100))
                    plan.append(dict(question_id=qid, question='Logic?', answer='A', voice='Jenny',
                        style=style, provider='azure', audio_path=name))
            (root/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plan))
            tasks, eligibility = run.recordings(root, bank)
            self.assertEqual(len(tasks), 4)
            self.assertEqual(eligibility['excluded_questions'], ['long'])
            self.assertEqual({r['condition'] for r in tasks}, set(run.STYLES))

    def test_wait_happens_before_generation_and_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            events = []
            with patch('sys.argv', ['test', '--output', d]), \
                 patch.object(run.subprocess, 'run', side_effect=lambda *a, **kw: events.append(('wait', a[0][-1]))), \
                 patch.object(run, 'generate', side_effect=lambda args: events.append(('generate',))), \
                 patch.object(run, 'evaluate', side_effect=lambda args: events.append(('evaluate',))):
                run.main()
            self.assertEqual(events, [('wait', 'svamp'), ('generate',), ('evaluate',)])

    def test_wait_detects_svamp_and_not_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'repo'; root.mkdir(); proc = Path(d)/'proc'; proc.mkdir()
            for pid, args in [(800, ['python', '-u', 'run_svamp500.py', '--mode', 'evaluate']),
                              (801, ['tail', '-f', 'outputs/svamp500.log'])]:
                folder = proc/str(pid); folder.mkdir(); (folder/'cwd').symlink_to(root)
                (folder/'stat').write_text(f'{pid} (python) S 0 0')
                (folder/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args))
            self.assertEqual(active_jobs('svamp', root, proc), [800])


if __name__ == '__main__':
    unittest.main()
