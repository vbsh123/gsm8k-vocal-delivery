from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import evaluate_bbh_audio_models as run


class BBHAudioModelTests(unittest.TestCase):
    def test_regeneration_uses_all_250_same_seeded_questions_and_only_neutral_angry(self):
        example = dict(input='Three birds.\nOptions:\n(A) Red\n(B) Blue\n(C) Green', target='(A)')
        data = json.dumps(dict(examples=[example]*250)).encode()
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            source = Path(d)
            (source/'logical_deduction_three_objects-source.json').write_bytes(data)
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()), \
                 patch('run_reasoning_transfer.generate_audio') as generate:
                run.generate_audio(source)
            self.assertEqual(generate.call_args.args, (source, run.select_questions(data, 250), ['neutral', 'angry']))
            self.assertEqual(len(generate.call_args.args[1]), 250)
            self.assertEqual((source/'logical_deduction_three_objects-source.json').read_bytes(), data)

    def fixture(self, root):
        source = root/'audio'; source.mkdir()
        example = dict(input='Red is left of blue.\nOptions:\n(A) Red\n(B) Blue\n(C) Green', target='(A)')
        data = json.dumps(dict(examples=[example]*4)).encode()
        (source/'logical_deduction_three_objects-source.json').write_bytes(data)
        bank = run.select_questions(data, 4)
        (source/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in bank))
        rows = []
        for q in bank:
            for style in ('neutral', 'angry', 'happy', 'sad'):
                path = source/f'{q["question_id"]}-{style}.wav'
                with wave.open(str(path), 'wb') as audio:
                    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                    audio.writeframes(b'\0\0'*(4000 if style == 'angry' else 3500))
                rows.append(dict(question_id=q['question_id'], provider='azure', voice='Jenny', style=style,
                                 question=q['question'], answer=q['answer'], audio_path=path.name))
        (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return source, hashlib.sha256(data).hexdigest()

    def test_same_original_questions_prompt_and_full_long_audio_without_extra_styles(self):
        with tempfile.TemporaryDirectory() as d:
            source, checksum = self.fixture(Path(d))
            with patch.object(run, 'SOURCE_SHA256', checksum):
                plan = run.make_plan(source, 'moss')
            self.assertEqual(len(plan['recordings']), 8)
            self.assertEqual({r['seconds'] for r in plan['recordings']}, {35., 40.})
            self.assertEqual({r['style'] for r in plan['recordings']}, {'neutral', 'angry'})
            self.assertEqual(plan['prompt'], run.PROMPT)
            self.assertEqual(plan['interventions'], [])
            self.assertEqual((plan['seed'], plan['max_new_tokens'], plan['recipe']['do_sample']), (42, 512, False))
            self.assertNotIn('vector', plan)

    def test_seeded_generation_resume_smoke_limit_and_no_transcript_in_call(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, checksum = self.fixture(root); output = root/'out'; output.mkdir()
            with patch.object(run, 'SOURCE_SHA256', checksum):
                plan = run.make_plan(source, 'moss')
            calls = []
            def generate(task):
                self.assertNotIn('question', task)
                calls.append((task['question_id'], task['style'], torch.initial_seed(), random.random(), float(np.random.random())))
                return 'The answer is (A).', 7
            with patch.object(run, 'moss_generator', return_value=generate), patch.object(run, 'kimi_generator') as kimi:
                run.run(plan, output, 'moss', Path('src'), max_generations=2)
                run.run(plan, output, 'moss', Path('src'))
                run.run(plan, output, 'moss', Path('src'))
                kimi.assert_not_called()
            self.assertEqual(len(calls), 8)
            self.assertEqual(len({c[2:] for c in calls}), 1)
            self.assertEqual(calls[0][2], 42)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual(len(saved), 8)
            self.assertTrue(all(r['response'] == 'The answer is (A).' and r['interventions'] == [] for r in saved))

    def test_moss_uses_author_processor_full_waveform_and_decodes_only_new_tokens(self):
        import numpy as np
        import torch
        class Inputs(dict):
            def to(self, device):
                return self
        inputs = Inputs(input_ids=torch.tensor([[8, 99, 7]]), audio_data=torch.ones(1, 2, 4))
        class Processor:
            config = SimpleNamespace(mel_sr=16000)
            audio_token_id = 99
            def __call__(self, **kwargs):
                self.called = kwargs
                return inputs
            def decode(self, ids, **kwargs):
                self.decoded = ids.tolist()
                return 'The answer is (A).'
        processor = Processor()
        class Model:
            device = 'cpu'
            dtype = torch.bfloat16
            def eval(self):
                return self
            def generate(self, **kwargs):
                self.called = kwargs
                return torch.tensor([[8, 99, 7, 201, 202]])
        model = Model(); waveform = np.zeros(40*16000, dtype=np.float32)
        model_class = SimpleNamespace(from_pretrained=lambda *a, **kw: model)
        processor_class = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
        modules = {'moss_audio_io': SimpleNamespace(load_audio=lambda *a, **kw: waveform),
                   'src.modeling_moss_audio': SimpleNamespace(MossAudioModel=model_class),
                   'src.processing_moss_audio': SimpleNamespace(MossAudioProcessor=processor_class)}
        with patch.dict(sys.modules, modules), patch.object(run.subprocess, 'check_output', return_value=run.MOSS_SOURCE_REVISION+'\n'):
            generate = run.moss_generator(Path('external/MOSS-Audio'))
            response, tokens = generate(dict(audio_file='full.wav'))
        self.assertIs(processor.called['audios'][0], waveform)
        self.assertEqual(processor.called['text'], run.PROMPT)
        self.assertEqual(processor.decoded, [201, 202])
        self.assertEqual((response, tokens), ('The answer is (A).', 2))
        self.assertEqual(model.called['audio_input_mask'].tolist(), [[False, True, False]])
        self.assertFalse(model.called['do_sample'])
        self.assertEqual(model.called['max_new_tokens'], 512)
        self.assertEqual(model.called['audio_data'].dtype, torch.bfloat16)

    def test_saved_response_summary_uses_installed_harness_on_matched_pairs(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, checksum = self.fixture(root); output = root/'out'; output.mkdir()
            with patch.object(run, 'SOURCE_SHA256', checksum):
                plan = run.make_plan(source, 'moss')
            (output/'run_config.json').write_text(json.dumps(plan))
            rows = [dict(r, response='The answer is (A).' if r['style'] == 'angry' else 'The answer is (B).')
                    for r in plan['recordings']]
            # Last angry answer is absent: report only the three completed pairs.
            (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            run.report(output)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['paired']), 3)
            for result in summary['paired']:
                expected = (3, 0, 0) if result['metric'] == run.FORMAT_METRIC else (3, 3, 0)
                self.assertEqual((result['pairs'], result['wins'], result['losses']), expected)

    def test_format_diagnostic_uses_official_strict_rules_but_is_independent_of_gold(self):
        from score_bbh_harness import HarnessScorer
        responses = ['The answer is (A).', 'The answer is (B).', '(A)',
                     'The answer is (D).', 'No answer.', 'The answer is (A) because it is red.']
        doc = dict(input='Which?\nOptions:\n(A) Red\n(B) Blue\n(C) Green', target='(A)')
        with tempfile.TemporaryDirectory() as d:
            scorer = HarnessScorer(Path(d))
            self.assertEqual(run.strict_format_bits(scorer, responses, [doc]*len(responses)),
                             [True, True, False, False, False, False])
            values = scorer.score(responses, [doc]*len(responses))
            self.assertTrue(values['strict-match'][0]['correct'])
            self.assertFalse(values['strict-match'][1]['correct'])

    def test_kimi_uses_full_audio_path_fixed_prompt_and_existing_greedy_recipe(self):
        import torch
        calls = []
        class Model:
            alm = SimpleNamespace(eval=lambda: None)
            def generate(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return None, 'The answer is (C).'
        model = Model()
        pm = ModuleType('kimia_infer.api.prompt_manager'); pm.Glm4Tokenizer = lambda path: path
        api = ModuleType('kimia_infer.api'); api.prompt_manager = pm
        package = ModuleType('kimia_infer'); package.api = api
        messages = lambda path, prompt: [dict(role='user', message_type='text', content=prompt),
                                        dict(role='user', message_type='audio', content=str(path))]
        fake = SimpleNamespace(assets=lambda source: ('weights', 'tokenizer', lambda **kw: model), messages=messages)
        modules = {'evaluate_kimi_audio': fake, 'kimia_infer': package, 'kimia_infer.api': api,
                   'kimia_infer.api.prompt_manager': pm}
        with patch.dict(sys.modules, modules):
            generate = run.kimi_generator(Path('external/Kimi-Audio'))
            self.assertEqual(generate(dict(audio_file='/audio/full-40-seconds.wav')), ('The answer is (C).', None))
        provided, settings = calls[0]
        self.assertEqual(provided[0]['content'], run.PROMPT)
        self.assertEqual(provided[1]['content'], '/audio/full-40-seconds.wav')
        self.assertEqual((settings['output_type'], settings['max_new_tokens']), ('text', 512))
        self.assertEqual((settings['text_temperature'], settings['audio_temperature']), (0., 0.))
        self.assertEqual(settings['text_repetition_penalty'], 1.)


if __name__ == '__main__':
    unittest.main()
