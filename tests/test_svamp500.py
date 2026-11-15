from contextlib import redirect_stdout
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

import run_svamp500 as run


class RegexFilterFixture:
    """Offline regex-filter fixture: first/last match and invalid fallback."""
    def __init__(self, regex_pattern, group_select=0):
        self.pattern, self.group = regex_pattern, group_select

    def apply(self, responses, docs):
        def extract(text):
            matches = re.findall(self.pattern, text)
            if not matches:
                return '[invalid]'
            value = matches[self.group]
            return next((part for part in value if part), '[invalid]') if isinstance(value, tuple) else value.strip()
        return [[extract(text) for text in group] for group in responses]


class SVAMPTests(unittest.TestCase):
    def test_hash_strict_differs_from_flexible_numeric(self):
        cases = [
            ('Final answer: 51', '51', True, False),
            ('We get 51.\n#### 51', '51', True, True),
            ('#### 51.00', '51', True, False),
            ('#### 1,234', '1234', True, True),
            ('#### -5', '-5', True, True),
            ('The result is 51', '51', True, False),
            ('#### 4\n#### 5', '5', True, False),
        ]
        for response,gold,numeric,strict in cases:
            with self.subTest(response=response):
                values = run.score(response, gold, RegexFilterFixture)
                self.assertEqual(values['flexible_numeric'], numeric)
                self.assertEqual(values['harness_strict'], strict)

    def test_format_change_cannot_resume_original_generations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = SimpleNamespace(source=root, output=root, answer_format='final-answer')
            eligibility = dict(excluded_questions=[])
            with patch.object(run, 'recordings', return_value=([], eligibility)), patch.object(run, 'report'):
                run.evaluate(args, {}, None)
                args.answer_format = 'hashes'
                with self.assertRaisesRegex(ValueError, 'separate result folder'):
                    run.evaluate(args, {}, None)

    def test_hash_mode_passes_new_prompt_to_decoder_and_reuses_audio(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); audio = root/'neutral.wav'; audio.write_bytes(b'audio')
            task = dict(question_id='svamp_chal-1', voice='Jenny', style='neutral', condition='neutral',
                        audio_path='neutral.wav', audio_file=str(audio),
                        audio_sha256=run.hashlib.sha256(b'audio').hexdigest())
            sf = SimpleNamespace(read=MagicMock(return_value=([0.0], 16000)))
            np = SimpleNamespace(random=SimpleNamespace(seed=MagicMock()))
            shared = SimpleNamespace(load_model=MagicMock(return_value=(MagicMock(), 'tokenizer', 'features')),
                                     gen=MagicMock(return_value='#### 5'))
            with patch.dict('sys.modules', {'torch': MagicMock(), 'soundfile': sf,
                                           'numpy': np, 'extract_vectors': shared}):
                run.run(dict(vector=None, prompt=run.HASH_PROMPT), [task],
                        {'svamp_chal-1': dict(answer='5')}, root, RegexFilterFixture)
            self.assertEqual(shared.gen.call_args.args[-2:], (run.HASH_PROMPT, 512))
            sf.read.assert_called_once_with(audio, dtype='float32')
            saved = json.loads((root/'results.jsonl').read_text())
            self.assertEqual(saved['prompt'], run.HASH_PROMPT)
            self.assertTrue(saved['harness_strict'])
            self.assertNotIn('strict', saved)

    def test_report_uses_only_completed_pairs_for_every_metric(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            rows = [dict(question_id=q, voice='Jenny', condition=condition, response=response, reference_answer='5')
                    for q,condition,response in [('paired','neutral','#### 5'), ('paired','angry','#### 6'),
                                                 ('unpaired','neutral','#### 5')]]
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            run.report(root, {q:dict(answer='5') for q in ('paired','unpaired')}, RegexFilterFixture)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual([r['n'] for r in summary['accuracy']], [1,1])
            strict = next(r for r in summary['paired'] if r['metric'] == 'harness_strict')
            self.assertEqual((strict['pairs'], strict['wins'], strict['losses']), (1,0,1))

    def test_fixed_selection_preserves_official_body_question_and_gold(self):
        data = json.dumps([dict(ID=str(i), Body=f'Body {i}', Question='How many?', Answer=51.0,
                               Equation='76-25', Type='Subtraction') for i in range(1000)])
        first = run.select_questions(data)
        self.assertEqual(first, run.select_questions(data))
        self.assertEqual(len(first), 500)
        self.assertEqual(len({r['question_id'] for r in first}), 500)
        self.assertEqual(first[0]['question'], 'Body '+first[0]['source_id']+' How many?')
        self.assertEqual(first[0]['answer'], '51')
        self.assertNotIn('76-25', first[0]['question'])

    def test_matched_eligibility_includes_only_unsteered_neutral_and_angry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); tasks = []; bank = {}
            for q in ('short', 'long'):
                bank[q] = dict(question='Question?', answer='5')
                for style in ('neutral', 'angry'):
                    name = f'{q}-{style}.wav'
                    with wave.open(str(root/name), 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                        audio.writeframes(b'\0\0' * (3001 if q == 'long' and style == 'angry' else 100))
                    tasks.append(dict(question_id=q, voice='Jenny', style=style, provider='azure',
                        audio_path=name, question='Question?', answer='5'))
            (root/'plan.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
            eligible, metadata = run.recordings(root, bank)
            self.assertEqual(len(eligible), 2)
            self.assertEqual(metadata['excluded_questions'], ['long'])
            self.assertEqual([r['condition'] for r in eligible], ['neutral', 'angry'])

    def test_uses_actual_extracted_prediction_and_numeric_equivalence(self):
        class Filter:
            def __init__(self, **kw):
                self.kw = kw
            def apply(self, responses, docs):
                return [['51.00']]
        values = run.score('The result is 51.00', '51', Filter)
        self.assertTrue(values['flexible_numeric'])
        self.assertFalse(values['harness_flexible'])
        self.assertFalse(run.score('The result is 51.00', '52', Filter)['flexible_numeric'])

    def test_summarizing_does_not_wait_for_gpu_or_read_vector(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            (source/'questions.jsonl').write_text('')
            with patch('sys.argv', ['test', '--mode', 'summary', '--source', str(source), '--output', str(root/'out')]), \
                 patch.object(run, 'harness_filter'), patch.object(run, 'report') as report, \
                 patch.object(run.fcntl, 'flock') as lock:
                run.main()
            report.assert_called_once(); lock.assert_not_called()

    def test_inference_reuses_one_model_and_resets_seed_for_both_deliveries(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); audio = root/'neutral.wav'; audio.write_bytes(b'audio')
            bank = {'svamp_chal-1': dict(answer='5')}
            tasks = [dict(question_id='svamp_chal-1', voice='Jenny', style='neutral',
                audio_path='neutral.wav', audio_file=str(audio), condition=c,
                audio_sha256=run.hashlib.sha256(b'audio').hexdigest()) for c in ['neutral', 'angry']]
            torch = MagicMock()
            sf = SimpleNamespace(read=MagicMock(return_value=([0.0], 16000)))
            np = SimpleNamespace(random=SimpleNamespace(seed=MagicMock()))
            shared = SimpleNamespace(load_model=MagicMock(return_value=(MagicMock(), 'tokenizer', 'features')),
                gen=MagicMock(return_value='Final answer: 5'))
            class Filter:
                def __init__(self, **kwargs): pass
                def apply(self, *args): return [['5']]
            with patch.dict('sys.modules', {'torch': torch, 'soundfile': sf, 'numpy': np, 'extract_vectors': shared}):
                run.run(dict(prompt=run.PROMPT), tasks, bank, root, Filter)
            shared.load_model.assert_called_once()
            self.assertEqual(shared.gen.call_count, 2)
            for call in shared.gen.call_args_list:
                self.assertEqual(call.args[-2:], (run.PROMPT, 512))
            self.assertEqual(torch.manual_seed.call_args_list[0].args, (42,))
            self.assertEqual(torch.manual_seed.call_args_list[1].args, (42,))
            torch.load.assert_not_called()
            self.assertEqual(sf.read.call_args_list[0], sf.read.call_args_list[1])

    def test_default_run_does_not_load_vector(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/'source'; source.mkdir(); (source/'questions.jsonl').write_text('')
            with patch('sys.argv', ['test', '--mode', 'evaluate', '--source', str(source), '--output', str(Path(d)/'out')]), \
                 patch.object(run, 'harness_filter'), patch.object(run, 'evaluate') as evaluate:
                run.main()
            self.assertEqual(evaluate.call_args.args[1], {})


if __name__ == '__main__':
    unittest.main()
