from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_gsm_strength_sweep as run


class StrengthSweepTests(unittest.TestCase):
    def previous(self, root):
        audio = root/'neutral.wav'; audio.write_bytes(b'audio')
        base = dict(question_id='q', voice='Jenny', style='neutral', condition='neutral',
                    answer='10', audio_file=str(audio), audio_sha256=run.reasoning.digest(audio))
        return dict(task='gsm8k', prompt=run.reasoning.HASH_PROMPT, collection_prompt='collection',
            seed=42, max_new_tokens=512, decoder='fast_dllm', bank={'q': dict(answer='10')}, extraction=[],
            recordings=[base, dict(base, style='angry', condition='angry'), dict(base, condition=run.reasoning.STEERED)])

    def test_exact_strength_grid_and_reuse_zero_one_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); benchmark = root/'old'; benchmark.mkdir(); output = root/'new'; output.mkdir()
            previous = self.previous(root); plan = run.make_plan(previous)
            self.assertEqual([r['strength'] for r in plan['recordings']], [-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.])
            self.assertTrue(all(r['style'] == 'neutral' for r in plan['recordings']))
            rows = [dict(r, prompt=plan['prompt'], seed=42, max_new_tokens=512, decoder='fast_dllm',
                         reference_answer='10', response='#### 10',
                         interventions=[dict(layer=31, group='answer', strength=1)] if r['condition'] == run.reasoning.STEERED else [])
                    for r in previous['recordings']]
            (benchmark/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            run.reuse_answers(benchmark, output, plan); run.reuse_answers(benchmark, output, plan)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual([r['condition'] for r in saved], ['neutral', run.reasoning.STEERED])
            previous['prompt'] = 'Old Final answer prompt'
            with self.assertRaisesRegex(ValueError, 'aligned-prompt'):
                run.make_plan(previous)

    def test_real_hook_uses_each_signed_strength_and_resume_retains_answers(self):
        import torch
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); plan = run.make_plan(self.previous(root))
            vector = root/'style-vectors.pt'
            sums = torch.zeros(32, 4, 1); sums[31, 3, 0] = 2
            counts = torch.ones(32, dtype=torch.long)
            torch.save(dict(pair_mean_sums={'angry': sums}, pair_counts={'angry': counts},
                            groups=['acoustic', 'semantic', 'combined', 'answer'], prompt='collection',
                            max_new_tokens=512, mode='fixed'), vector)
            plan.update(vector=str(vector), vector_sha256=run.reasoning.digest(vector))
            class Block(torch.nn.Module):
                def forward(self, h):
                    return (h,)
            blocks = torch.nn.ModuleList([Block() for _ in range(32)])
            model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
            calls = []
            def gen(audio, sr, model, tokenizer, features, prompt, tokens, seconds):
                self.assertEqual((prompt, tokens, seconds, torch.initial_seed()), (plan['prompt'], 512, 30, 42))
                hidden = blocks[31](torch.full((1, 514, 1), 10.))[0]
                self.assertEqual(float(hidden[0, 0, 0]), 10.)
                calls.append(float(hidden[0, -1, 0]))
                return f'#### {calls[-1]:g}', dict(mode='native')
            class Scorer:
                def score(self, response, gold):
                    return {'flexible-numeric': response == '#### '+gold}, {'flexible-extract': response.split()[-1]}
            original_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return original_to(tensor, *args, **kwargs)
            with patch.dict(sys.modules, {'soundfile': SimpleNamespace(read=lambda *a, **kw: (None, 16000)),
                    'extract_vectors': SimpleNamespace(gen_chunked=gen, load_model=lambda: (model, None, None))}), \
                 patch.object(run.reasoning, 'scorer_for', return_value=Scorer()), \
                 patch.object(torch.Tensor, 'to', cpu_to), patch.dict(os.environ, {'HF_HOME': str(root/'cache')}):
                run.reasoning.evaluate(plan, root); run.reasoning.evaluate(plan, root)
            self.assertEqual(calls, [6., 7., 8., 9., 10., 11., 12., 13., 14.])
            rows = run.read_rows(root/'results.jsonl')
            self.assertEqual([r['steered_forwards'] for r in rows], [1, 1, 1, 1, 0, 1, 1, 1, 1])
            self.assertFalse(blocks[31]._forward_hooks)


if __name__ == '__main__':
    unittest.main()
