import json
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave
from unittest.mock import patch

import run_kimi_gsm8k as run


class KimiGSMTests(unittest.TestCase):
    def test_full_cohort_half_split_frozen_without_model_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/'audio'; output = Path(d)/'out'; output.mkdir()
            plan = run.select(source, output)
            self.assertEqual(plan, run.select(source, output))
            self.assertEqual((len(plan['collection']), len(plan['test'])), (634, 635))
            train = {q['question_id'] for q in plan['collection']}
            test = {q['question_id'] for q in plan['test']}
            self.assertFalse(train & test)
            self.assertEqual(train | test, {f'gsm8k_test_{i:05d}' for i in range(50, 1319)})
            self.assertEqual((plan['layer'], plan['group'], plan['strength']), (17, 'answer', 1.))
            self.assertIn('#### <number>', plan['prompt'])

    def test_shared_evaluation_uses_gsm_condition_and_separate_vector(self):
        from tests.test_kimi_multiarith import KimiMultiArithTests
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = KimiMultiArithTests().fixture(root)
            plan.update(protocol=run.PROTOCOL, steered_condition=run.STEERED)
            vector = dict(collection_question_ids=['train0', 'train1'], answer_direction='gsm vector')
            model = SimpleNamespace(prompt_manager=SimpleNamespace(text_tokenizer=SimpleNamespace(
                encode=lambda s, *, bos, eos: list(s))))
            with patch.object(run.shared.kimi, 'digest', side_effect=lambda p: str(p)), \
                 patch.object(run.shared.kimi, 'generate', return_value='#### 2'), \
                 patch.object(run.shared.kimi, 'generate_steered', return_value='#### 2') as steer:
                run.shared.evaluate(plan, root, model, vector)
            rows = run.shared.read_rows(root/'results.jsonl')
            self.assertEqual({r['condition'] for r in rows}, {'neutral', 'angry', run.STEERED})
            self.assertEqual(steer.call_args.args[3:], (17, 'answer', 'gsm vector'))

    def test_preparation_preserves_full_cohort_and_gsm_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'audio'; output = root/'out'; output.mkdir()
            selection = run.select(source, output)
            with wave.open(str(source/'clip.wav'), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(b'\0\0'*160)
            rows = [dict(q, provider='azure', voice='Jenny', style=s, audio_path='clip.wav')
                    for q in selection['collection']+selection['test'] for s in ('neutral', 'angry')]
            (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args = SimpleNamespace(source=source, output=output)
            # Preparation imports only these pins, not inference; this CPU environment
            # deliberately lacks the separate Kimi/Qwen audio dependencies.
            tree = ast.parse(Path(run.__file__).with_name('evaluate_kimi_audio.py').read_text())
            names = {'MODEL', 'REVISION', 'SOURCE_REVISION', 'TOKENIZER_MODEL', 'TOKENIZER_REVISION'}
            pins = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}
            with patch.dict(sys.modules, {'evaluate_kimi_audio': SimpleNamespace(**pins)}):
                plan = run.shared.prepare(args, selection, dataset='GSM8K', steered_condition=run.STEERED)
                self.assertEqual(plan['protocol'], run.PROTOCOL)
                self.assertEqual(plan['steered_condition'], run.STEERED)
                self.assertEqual((len(plan['collection']), len(plan['test'])), (634, 635))
                self.assertEqual(plan, run.shared.prepare(args, selection, dataset='GSM8K', steered_condition=run.STEERED))


if __name__ == '__main__':
    unittest.main()
