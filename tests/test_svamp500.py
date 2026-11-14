from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

import run_svamp500 as run


class SVAMPTests(unittest.TestCase):
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

    def test_matched_eligibility_and_steering_uses_same_neutral_audio(self):
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
            eligible, metadata = run.recordings(root, bank, True)
            self.assertEqual(len(eligible), 3)
            self.assertEqual(metadata['excluded_questions'], ['long'])
            neutral, angry, steering = eligible
            self.assertEqual(neutral['audio_sha256'], steering['audio_sha256'])
            self.assertEqual(neutral['audio_file'], steering['audio_file'])
            self.assertEqual(steering['condition'], run.STEERED)
            self.assertEqual(steering['style'], 'neutral')

    def test_uses_actual_extracted_prediction_and_numeric_equivalence(self):
        class Filter:
            def __init__(self, **kw):
                self.kw = kw
            def apply(self, responses, docs):
                return [['51.00']]
        values = run.score('The result is 51.00', '51', Filter)
        self.assertTrue(values['flexible_numeric'])
        self.assertFalse(values['harness_flexible'])
        self.assertFalse(values['strict'])
        self.assertFalse(run.score('The result is 51.00', '52', Filter)['flexible_numeric'])

    def test_summarizing_does_not_wait_for_gpu_or_read_vector(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            (source/'questions.jsonl').write_text('')
            with patch('sys.argv', ['test', '--mode', 'summary', '--source', str(source), '--output', str(root/'out')]), \
                 patch.object(run, 'harness_filter'), patch.object(run, 'report') as report, \
                 patch.object(run.fcntl, 'flock') as lock, patch.object(run, 'find_vectors') as vectors:
                run.main()
            report.assert_called_once(); lock.assert_not_called(); vectors.assert_not_called()

    def test_original_vector_discovery_from_existing_winner_config(self):
        with tempfile.TemporaryDirectory() as d:
            vector = Path(d)/'original.pt'; vector.touch()
            self.assertEqual(run.find_vectors(vector), vector.absolute())
            with self.assertRaises(FileNotFoundError):
                run.find_vectors(Path(d)/'missing.pt')

    def test_inference_reuses_one_model_same_prompt_seed_and_neutral_audio(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); audio = root/'neutral.wav'; audio.write_bytes(b'audio')
            bank = {'svamp_chal-1': dict(answer='5')}
            tasks = [dict(question_id='svamp_chal-1', voice='Jenny', style='neutral',
                audio_path='neutral.wav', audio_file=str(audio), condition=c,
                audio_sha256=run.hashlib.sha256(b'audio').hexdigest()) for c in ['neutral', run.STEERED]]
            state = dict(prompt=run.PROMPT, max_new_tokens=512, mode='fixed', groups=['answer'],
                pair_counts={'angry': {31: 425}}, pair_mean_sums={'angry': MagicMock()})
            torch = MagicMock(); torch.load.return_value = state
            sf = SimpleNamespace(read=MagicMock(return_value=([0.0], 16000)))
            np = SimpleNamespace(random=SimpleNamespace(seed=MagicMock()))
            shared = SimpleNamespace(load_model=MagicMock(return_value=(MagicMock(), 'tokenizer', 'features')),
                gen=MagicMock(return_value='Final answer: 5'))
            class Filter:
                def __init__(self, **kwargs): pass
                def apply(self, *args): return [['5']]
            with patch.dict('sys.modules', {'torch': torch, 'soundfile': sf, 'numpy': np, 'extract_vectors': shared}), \
                 patch.object(run, 'install_interventions') as interventions:
                run.run(dict(vector=dict(path='original.pt')), tasks, bank, root, Filter)
            shared.load_model.assert_called_once()
            self.assertEqual(shared.gen.call_count, 2)
            for call in shared.gen.call_args_list:
                self.assertEqual(call.args[-2:], (run.PROMPT, 512))
            self.assertEqual(torch.manual_seed.call_args_list[0].args, (42,))
            self.assertEqual(torch.manual_seed.call_args_list[1].args, (42,))
            self.assertEqual(interventions.call_args_list[0].args[2], [])
            self.assertEqual(interventions.call_args_list[1].args[2], [dict(layer=31, group='answer', strength=1)])
            self.assertEqual(sf.read.call_args_list[0], sf.read.call_args_list[1])


if __name__ == '__main__':
    unittest.main()
