from contextlib import redirect_stdout
import io
import json
import os
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
            full, chunked_eligibility = run.recordings(root, bank, allow_long=True)
            self.assertEqual(len(full), 8)
            self.assertEqual(chunked_eligibility['excluded_questions'], [])
            self.assertTrue(any(t['seconds'] > 30 for t in full))

    def test_chunked_smoke_uses_separate_output_and_does_not_generate_audio(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'smoke'
            with patch('sys.argv', ['test', '--mode', 'evaluate', '--chunked-audio', '--smoke-test', '--output', str(output)]), \
                 patch.object(run.subprocess, 'run'), patch.object(run, 'evaluate') as evaluate, \
                 patch.object(run, 'generate') as generate:
                run.main()
            generate.assert_not_called()
            args = evaluate.call_args.args[0]
            self.assertTrue(args.chunked_audio)
            self.assertTrue(args.smoke_test)
            self.assertEqual(args.chunk_seconds, 30)

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

    def test_default_summary_calls_upstream_scorer_without_generation(self):
        from types import SimpleNamespace
        with patch('score_bbh_harness.report_saved') as report:
            run.summarize(SimpleNamespace(output=Path('answers'), source=Path('audio')))
        report.assert_called_once_with(Path('answers'), Path('audio'))


@unittest.skipUnless(os.environ.get('BBH_HARNESS_CACHE'), 'Set BBH_HARNESS_CACHE to a pinned upstream cache; no network during tests')
class UpstreamBBHTests(unittest.TestCase):
    def test_actual_upstream_strict_and_flexible_keep_their_format_rules(self):
        from score_bbh_harness import HarnessScorer
        scorer = HarnessScorer(os.environ['BBH_HARNESS_CACHE'])
        docs = [dict(input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green', target='(B)')]*4
        values = scorer.score(['The answer is (B).', 'Final answer: B', 'No answer supplied.', 'The answer is (A).'], docs)
        self.assertEqual([r['correct'] for r in values['strict-match']], [True,False,False,False])
        self.assertEqual([r['correct'] for r in values['flexible-extract']], [True,True,False,False])

    def test_report_requires_all_four_deliveries_and_preserves_saved_answers(self):
        from score_bbh_harness import report_saved
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source = root/'source'; source.mkdir()
            official = dict(input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green',target='(B)')
            payload = json.dumps(dict(examples=[official])).encode()
            (source/'logical_deduction_three_objects-source.json').write_bytes(payload)
            question = dict(question_id='q', source_index=0, original_input=official['input'],
                            original_target='(B)',answer='B')
            (source/'questions.jsonl').write_text(json.dumps(question)+'\n')
            rows = [dict(question_id='q',voice=voice,condition=style,reference_answer='B',
                         response='The answer is (A).' if style == 'angry' else 'The answer is (B).')
                    for voice,styles in [('matched',run.STYLES),('incomplete',['neutral'])] for style in styles]
            saved = ''.join(json.dumps(r)+'\n' for r in rows)
            (root/'results.jsonl').write_text(saved)
            with patch.object(run, 'SOURCE_SHA256', run.hashlib.sha256(payload).hexdigest()):
                summary = report_saved(root,source,cache=os.environ['BBH_HARNESS_CACHE'])
            self.assertEqual(summary['matched_questions'],1)
            self.assertEqual(summary['unmatched_rows_excluded'],1)
            self.assertTrue(all(r['n'] == 1 for r in summary['accuracy']))
            strict = next(r for r in summary['paired'] if r['metric']=='strict-match' and r['condition']=='angry')
            self.assertEqual((strict['wins'],strict['losses']),(0,1))
            self.assertEqual((root/'results.jsonl').read_text(),saved)


if __name__ == '__main__':
    unittest.main()
