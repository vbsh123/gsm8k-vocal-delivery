import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import steer_bbh_mean as run


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


class BBHSteeringTests(unittest.TestCase):
    def source(self, root):
        questions = [dict(question_id=f'q{i}', question=f'Question {i}', answer='A',
            choices=[dict(label=letter, text=letter) for letter in 'ABC']) for i in range(250)]
        write_rows(root/'questions.jsonl', questions)
        plan = []
        for q in questions:
            for style in ('neutral', 'angry'):
                path = root/f'{q["question_id"]}-{style}.wav'
                with wave.open(str(path), 'wb') as audio:
                    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                    audio.writeframes(b'\0\0'*4000)
                plan.append(dict(q, provider='azure', voice='Jenny', style=style, audio_path=path.name))
        write_rows(root/'plan.jsonl', plan)

    def test_fixed_split_disjoint_long_audio_preserved_and_standard_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.source(root)
            plan = run.make_plan(root)
            self.assertEqual(plan, run.make_plan(root))
            train = {p['question_id'] for p in plan['extraction']}
            test = {p['question_id'] for p in plan['test']}
            self.assertEqual((len(train), len(test), len(train & test)), (125, 125, 0))
            self.assertEqual(plan['test'][0]['question_id'], 'q125')
            self.assertEqual(plan['test'][0]['angry']['seconds'], 40)
            self.assertEqual((plan['layer'], plan['group'], plan['strength']), (31, 'answer', 1))
            self.assertEqual(plan['prompt'], run.PROMPT)
            self.assertEqual(plan['audio_encoding']['chunk_seconds'], 30)
            self.assertEqual((plan['seed'], plan['max_new_tokens'], plan['decoder']), (42, 512, 'fast_dllm'))

    def test_missing_angry_pair_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.source(root)
            rows = run.read_rows(root/'plan.jsonl')
            write_rows(root/'plan.jsonl', rows[:-1])
            with self.assertRaisesRegex(ValueError, '250 neutral/angry pairs'):
                run.make_plan(root)

    def test_reuse_native_short_clip_under_chunked_recipe_and_no_extraction_leak(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); benchmark = root/'benchmark'; benchmark.mkdir()
            output = root/'out'; output.mkdir()
            task = dict(voice='Jenny', answer='A', audio_sha256='expected')
            plan = dict(test=[dict(question_id='test', neutral=task, angry=task)])
            recipe = dict(seed=42, prompt=run.PROMPT, max_new_tokens=512, decoder='fast_dllm',
                audio_encoding=dict(mode='chunked-whisper-global-adapters-v1', chunk_seconds=30))
            config = dict(recipe, recordings=[dict(question_id=q, voice='Jenny', style='neutral',
                audio_sha256='expected') for q in ('test', 'train')])
            run.write_json(benchmark/'run_config.json', config)
            rows = [dict(recipe, question_id=q, voice='Jenny', condition='neutral', reference_answer='A',
                response='The answer is (A).', audio_encoding=dict(mode='native', chunk_seconds=30))
                for q in ('test', 'train')]
            write_rows(benchmark/'results.jsonl', rows)
            args = SimpleNamespace(benchmark=benchmark, output=output)
            run.reuse_answers(args, plan); run.reuse_answers(args, plan)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]['question_id'], 'test')
            self.assertEqual(saved[0]['response'], rows[0]['response'])
            config['recordings'][0]['audio_sha256'] = 'changed'
            run.write_json(benchmark/'run_config.json', config)
            (output/'results.jsonl').unlink()
            with self.assertRaisesRegex(ValueError, 'answer/audio differs'):
                run.reuse_answers(args, plan)

    def test_legacy_prompt_cannot_reuse_answers(self):
        row = dict(seed=42, prompt='Final answer: A', max_new_tokens=512, decoder='fast_dllm',
            audio_encoding=dict(mode='chunked-whisper-global-adapters-v1', chunk_seconds=30))
        self.assertFalse(run.compatible(row))

    def test_collection_matches_forwards_then_questions_and_steering_uses_same_decoder(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            output = root/'out'; output.mkdir(); (output/'pairs').mkdir()
            self.source(source); plan = run.make_plan(source)
            for i, pair in enumerate(plan['extraction'][1:], 1):
                torch.save(dict(question_id=pair['question_id'], difference=torch.tensor([2.])), output/'pairs'/f'{i:05d}.pt')
            write_rows(output/'results.jsonl', [dict(question_id=p['question_id'], voice='Jenny', condition=c)
                for p in plan['test'] for c in run.CONDITIONS
                if (p['question_id'], c) != ('q125', run.CONDITION)])
            class Block(torch.nn.Module):
                def forward(self, hidden):
                    return (hidden,)
            blocks = torch.nn.ModuleList([Block() for _ in range(32)])
            model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
            calls = []
            def gen(audio, sr, received_model, tokenizer, features, prompt, tokens, chunk_seconds):
                self.assertIs(received_model, model)
                self.assertEqual((sr, prompt, tokens, chunk_seconds, torch.initial_seed()), (16000, run.PROMPT, 512, 30, 42))
                values = [5., 9., 999.] if audio[0] else [1., 3.]
                final = None
                for value in values:
                    h = torch.full((1, 514, 1), value)
                    final = blocks[31](h)[0]
                    self.assertEqual(final[0, 0, 0].item(), value)
                calls.append(final[0, -1, 0].item())
                return 'The answer is (A).', dict(mode='chunked-whisper-global-adapters-v1', chunk_seconds=30)
            def read(path, **kwargs):
                return np.array([int('angry' in path.name)], dtype=np.float32), 16000
            original_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return original_to(tensor, *args, **kwargs)
            args = SimpleNamespace(output=output, source=source)
            with patch.dict(sys.modules, {'soundfile': SimpleNamespace(read=read),
                    'extract_vectors': SimpleNamespace(gen_chunked=gen, load_model=lambda: (model, None, None))}), \
                 patch.object(torch.Tensor, 'to', cpu_to), patch.dict(run.os.environ, {'HF_HOME': str(root/'cache')}):
                run.run(args, plan)
            state = torch.load(output/'style-vectors.pt', weights_only=True)
            # ((5-1)+(9-3))/2 = 5; 999 is beyond the matched forwards.
            expected = (5 + 124*2)/125
            self.assertAlmostEqual((state['pair_mean_sums']['angry'][31, 3]/125).item(), expected, places=5)
            self.assertAlmostEqual(calls[-1], 3+expected, delta=.02)
            self.assertEqual(len(calls), 3)
            saved = run.read_rows(output/'results.jsonl')[-1]
            self.assertEqual((saved['condition'], saved['prompt']), (run.CONDITION, run.PROMPT))
            self.assertFalse(blocks[31]._forward_hooks)

    def test_prepare_links_old_audio_and_never_edits_old_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = root/'old'; old.mkdir(); new = root/'new'
            data = b'original source'
            (old/'logical_deduction_three_objects-source.json').write_bytes(data)
            task = dict(question_id='q', voice='Jenny', style='neutral', provider='azure',
                question='Question', answer='A', request='same SSML', audio_path='old.wav')
            write_rows(old/'plan.jsonl', [task]); (old/'old.wav').write_bytes(b'audio')
            original_plan = (old/'plan.jsonl').read_bytes()
            args = SimpleNamespace(existing_source=old, source=new, mode='prepare')
            def cloud(command, **kwargs):
                self.assertIn('--prepare-only', command)
                write_rows(new/'plan.jsonl', [dict(task, audio_path='new.wav')])
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()), \
                 patch.object(run, 'select_questions', return_value=[dict(question_id='q')]), \
                 patch.object(run.subprocess, 'run', side_effect=cloud):
                run.prepare(args)
            self.assertTrue((new/'new.wav').is_symlink())
            self.assertEqual((new/'new.wav').read_bytes(), b'audio')
            self.assertEqual((old/'plan.jsonl').read_bytes(), original_plan)


if __name__ == '__main__':
    unittest.main()
