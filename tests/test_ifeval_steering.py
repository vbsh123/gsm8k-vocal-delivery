from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import steer_ifeval_mean as run


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


class IFEvalSteeringTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'; source.mkdir(); benchmark = root/'old'; benchmark.mkdir()
        questions = [dict(question_id=f'q{i}', key=i, question='Write two bullets.', prompt='Write two bullets.',
                          instruction_id_list=['bullets'], kwargs=[{}]) for i in range(4)]
        write_rows(source/'questions.jsonl', questions)
        tasks = []
        for q in questions:
            for style in run.ifeval.STYLES:
                path = source/f'{q["question_id"]}-{style}.wav'
                with wave.open(str(path), 'wb') as audio:
                    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
                    audio.writeframes(b'\0\0'*160)
                tasks.append(dict(question_id=q['question_id'], voice='Jenny', style=style,
                                  audio_path=path.name, audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        previous = dict(recordings=tasks, prompt=run.ifeval.PROMPT, seed=42, max_new_tokens=1024,
                        decoder='fast_dllm', revision=run.ifeval.REVISION)
        (benchmark/'run_config.json').write_text(json.dumps(previous))
        write_rows(benchmark/'results.jsonl', [dict(t, response='Original answer.', wrapper=run.ifeval.PROMPT,
                                                  seed=42, max_new_tokens=1024) for t in tasks])
        return source, benchmark

    def test_split_and_reuse_preserve_cohort_and_original_budget_without_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, benchmark = self.fixture(root); output = root/'new'; output.mkdir()
            plan = run.make_plan(source, benchmark)
            self.assertEqual(plan, run.make_plan(source, benchmark))
            train = {p['question_id'] for p in plan['extraction']}; test = {p['question_id'] for p in plan['test']}
            self.assertEqual((len(train), len(test), len(train & test)), (2, 2, 0))
            self.assertEqual((plan['prompt'], plan['max_new_tokens'], plan['strength']), (run.ifeval.PROMPT, 1024, 1))
            run.reuse_answers(benchmark, output, plan); run.reuse_answers(benchmark, output, plan)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual(len(saved), 4)
            self.assertEqual({r['question_id'] for r in saved}, test)
            self.assertEqual({r['condition'] for r in saved}, {'neutral', 'angry'})

    def test_shared_collection_uses_1024_answer_positions_actual_pair_count_and_custom_condition(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, benchmark = self.fixture(root); output = root/'new'; output.mkdir()
            plan = run.make_plan(source, benchmark); run.reuse_answers(benchmark, output, plan)
            class Block(torch.nn.Module):
                def forward(self, hidden):
                    return (hidden,)
            blocks = torch.nn.ModuleList([Block() for _ in range(32)])
            model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3])
            calls = []
            def gen(audio, sr, received_model, tok, features, prompt, tokens, seconds):
                self.assertEqual((prompt, tokens, seconds, torch.initial_seed()), (run.ifeval.PROMPT, 1024, 30, 42))
                for value in ([5., 9., 999.] if audio[0] else [1., 3.]):
                    hidden = blocks[31](torch.full((1, 1026, 1), value))[0]
                    self.assertEqual(hidden[0, 0, 0].item(), value)
                calls.append(hidden[0, -1, 0].item())
                return 'Two bullets.', dict(mode='native', chunk_seconds=30)
            def read(path, **kwargs):
                return np.array([int('angry' in path.name)], dtype=np.float32), 16000
            original_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return original_to(tensor, *args, **kwargs)
            with patch.dict(sys.modules, {'soundfile': SimpleNamespace(read=read),
                    'extract_vectors': SimpleNamespace(gen_chunked=gen, load_model=lambda: (model, tokenizer, None))}), \
                 patch.object(torch.Tensor, 'to', cpu_to), patch.dict(run.collection.os.environ, {'HF_HOME': str(root/'cache')}):
                run.collection.run(SimpleNamespace(source=source, output=output), plan)
            state = torch.load(output/'style-vectors.pt', weights_only=True)
            self.assertEqual(int(state['pair_counts']['angry'][31]), 2)
            self.assertEqual(state['max_new_tokens'], 1024)
            self.assertEqual(float(state['pair_mean_sums']['angry'][31, 3]/2), 5.)
            saved = run.read_rows(output/'results.jsonl')
            steered = [r for r in saved if r['condition'] == run.CONDITION]
            self.assertEqual(len(steered), 2)
            self.assertTrue(all(r['response_tokens'] == 3 for r in steered))
            self.assertEqual(calls[-2:], [8., 8.])
            self.assertFalse(blocks[31]._forward_hooks)

    def test_official_report_keeps_steered_neutral_separate_from_baseline_neutral(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); source, _ = self.fixture(root); output = root/'new'; output.mkdir()
            write_rows(output/'results.jsonl', [dict(question_id='q0', voice='Jenny', style='neutral',
                condition=c, response='ok' if c == run.CONDITION else 'bad') for c in ('neutral', run.CONDITION)])
            def score(example, response, evaluator):
                good = response == 'ok'
                return {m: dict(prompt=good, instructions=[good]) for m in ('strict', 'loose')}
            with patch.object(run.ifeval, 'score', side_effect=score):
                run.ifeval.report(SimpleNamespace(source=source, output=output), styles=['neutral', run.CONDITION],
                                  evaluator=object(), condition_field='condition')
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual([r['strict_prompt'] for r in summary['totals']], [0., 1.])
            self.assertEqual(summary['paired_prompt_comparisons'][0]['wins'], 1)


if __name__ == '__main__':
    unittest.main()
