import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import evaluate_bbh5_candidates as run


class BBHFiveCandidatesTests(unittest.TestCase):
    def fixture(self, root):
        from tests.test_kimi_bbh5 import KimiBBH5Tests
        args, data = KimiBBH5Tests().fixture(root)
        questions = run.reasoning.bbh_questions(data, 'five')
        (args.source/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
        with wave.open(str(args.source/'clip.wav'), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(100)
            f.writeframes(b'\0\0'*4700)
        rows = [dict(q, provider='azure', voice='Jenny', style=s, audio_path='clip.wav')
                for q in questions for s in ('neutral', 'angry')]
        (args.source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return args, data

    def test_full_250_both_prompts_five_choices_and_no_steering(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d))
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                old = run.make_plan(args.source, 'phi4', 'original')
                new = run.make_plan(args.source, 'phi4', 'explicit')
            self.assertEqual(len(old['bank']), 250)
            self.assertEqual(len(old['recordings']), 500)
            self.assertEqual(old['recordings'], new['recordings'])
            self.assertEqual(old['interventions'], [])
            self.assertNotIn('vector', old)
            self.assertTrue(all(r['seconds'] == 47 for r in old['recordings']))
            self.assertIn('five choices A, B, C, D, E', old['prompt'])
            self.assertEqual(new['prompt'], old['prompt'].replace("Let's think step by step.",
                'Give concise step-by-step reasoning with only the necessary deductions.'))

    def test_audio_reuse_makes_no_tts_requests(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d)); args.model = 'phi4'
            before = (args.source/'plan.jsonl').read_bytes()
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()), \
                 patch.object(run, 'prepare_audio') as tts:
                run.audio(args)
            tts.assert_not_called()
            self.assertEqual(before, (args.source/'plan.jsonl').read_bytes())

    def test_native_five_scorer_accepts_d_and_e_and_preserves_raw_answers(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d))
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                plan = run.make_plan(args.source, 'phi4', 'original')
            out = args.output
            (out/'run_config.json').write_text(json.dumps(plan))
            rows = [dict(t, prompt=plan['prompt'], response_tokens=6,
                response='The answer is (E).' if t['style'] == 'angry' else 'The answer is (D).')
                for t in plan['recordings'][:2]]
            path = out/'results.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); before = path.read_bytes()
            run.bbh.report(out, objects='five')
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['scorer']['task'], 'bbh_cot_zeroshot_logical_deduction_five_objects')
            formats = [r for r in summary['accuracy'] if r['metric'] == 'strict-format-extractable']
            self.assertEqual([r['correct'] for r in formats], [1, 1])
            self.assertEqual({r['cohort'] for r in summary['accuracy']}, {'all'})
            self.assertEqual(before, path.read_bytes())


if __name__ == '__main__': unittest.main()
