import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import evaluate_gsm_candidates as run


class GSMCandidateTests(unittest.TestCase):
    def test_flamingo_fixed400_same_cohort_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root); before=(source/'plan.jsonl').read_bytes()
            args=SimpleNamespace(source=source,model='flamingo',prompt='original',output=root/'out',start_index=0,num_questions=400)
            plans=run.prepare(args);self.assertEqual(len(plans),1)
            output,plan=plans[0]
            self.assertEqual(len(plan['bank']),400);self.assertEqual(len(plan['recordings']),800)
            self.assertEqual(set(plan['bank']),{q['question_id'] for q in run.questions()[:400]})
            self.assertEqual(plan['prompt'],run.ORIGINAL);self.assertEqual(plan['interventions'],[])
            self.assertEqual(before,(source/'plan.jsonl').read_bytes())
            self.assertEqual(run.prepare(args)[0][1],plan)

    def fixture(self, root):
        source = root/'audio'; source.mkdir()
        rows = run.questions()
        (source/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with wave.open(str(source/'audio.wav'), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(100)
            f.writeframes(b'\0\0'*4500)
        tasks = [dict(q, provider='azure', voice='Jenny', style=s, audio_path='audio.wav')
                 for q in rows for s in ('neutral', 'angry')]
        (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
        return source

    def test_all_1269_no_split_steering_or_long_audio_filter_and_aligned_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(Path(d))
            old = run.make_plan(source, 'phi4', 'original')
            new = run.make_plan(source, 'phi4', 'explicit')
            self.assertEqual(len(old['bank']), 1269)
            self.assertEqual(len(old['recordings']), 2538)
            self.assertEqual(old['recordings'], new['recordings'])
            self.assertTrue(all(r['seconds'] == 45 for r in old['recordings']))
            self.assertEqual(old['interventions'], [])
            self.assertNotIn('vector', old)
            self.assertIn('#### <number>', old['prompt'])
            self.assertEqual(new['prompt'], run.reasoning.HASH_PROMPT)
            self.assertEqual(new['prompt'].replace('Give concise step-by-step reasoning with only the necessary calculations.',
                "Let's think step by step."), old['prompt'])
            self.assertEqual(old['seed'], 42)
            self.assertEqual(old['max_new_tokens'], 512)

    def test_audio_reuses_existing_kimi_clips_without_tts(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(Path(d)); before = (source/'plan.jsonl').read_bytes()
            args = SimpleNamespace(source=source, model='phi4')
            with patch.object(run, 'prepare_audio') as tts:
                run.audio(args)
            tts.assert_not_called()
            self.assertEqual(before, (source/'plan.jsonl').read_bytes())

    def test_saved_answer_scoring_distinguishes_correctness_from_format(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = self.fixture(root)
            plan = run.make_plan(source, 'phi4', 'explicit')
            out = root/'out'; out.mkdir()
            (out/'run_config.json').write_text(json.dumps(plan))
            pair = plan['recordings'][:2]; gold = pair[0]['reference_answer']
            rows = [dict(task, prompt=plan['prompt'], response_tokens=10,
                         response=f'#### {gold}' if task['style'] == 'angry' else f'The answer is {gold}.') for task in pair]
            path = out/'results.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); before = path.read_bytes()
            run.report(out)
            summary = json.loads((out/'summary.json').read_text())
            by_style = {r['style']: r for r in summary['accuracy']}
            self.assertEqual(by_style['neutral']['correct']['flexible-numeric'], 1)
            self.assertEqual(by_style['neutral']['correct']['strict-match'], 0)
            self.assertEqual(by_style['angry']['correct']['strict-match'], 1)
            self.assertEqual(summary['primary_metric'], 'flexible-numeric')
            self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
