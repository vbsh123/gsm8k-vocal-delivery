import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_gsm_vanilla as vanilla


class VanillaTests(unittest.TestCase):
    def registry(self, path_prefix='/first'):
        ids = [f'q{i:02d}' for i in range(16)]
        options, labels = {}, {}
        for q in ids:
            for style in vanilla.STYLES:
                record = dict(question_id=q, voice='en-US-JennyNeural', style=style,
                              audio_sha256=q+style, audio_file=path_prefix+'/'+q+style,
                              reference_answer='2', seconds=1.)
                options['gsm8k', q, style] = [dict(record=record, question='Question '+q, answer='2')]
                labels[q+style] = dict(classification_status='classified', emotion=style)
        # Distinct per-emotion exclusions. Neutral recognition is deliberately wrong.
        for i, e in enumerate(vanilla.EMOTIONS):
            labels[ids[i]+e]['emotion'] = 'other'
        for q in ids:
            labels[q+'neutral']['emotion'] = 'other'
        return SimpleNamespace(neutral_ids=lambda name: ids,
            choose=lambda name, q, s, **kw: options[name, q, s][0], options=options, labels=labels)

    def prepare(self, root, prefix='/first'):
        return vanilla.prepare(self.registry(prefix), root, expected_pool=16, test_count=4, fit_count=5)

    def test_selection_is_label_blind_for_test_and_independent_for_each_emotion(self):
        pool = [f'q{i:03}' for i in range(100)]
        all_agreed = {e: set(pool) for e in vanilla.EMOTIONS}
        base = vanilla.select(pool, all_agreed, 20, 25)
        agreed = {e: set(pool)-set(base['fitting_order'][i::3]) for i, e in enumerate(vanilla.EMOTIONS)}
        result = vanilla.select(pool[::-1], agreed, 20, 25)
        self.assertEqual(result['test_ids'], base['test_ids'])
        self.assertEqual(result['collections']['angry']['random'], base['collections']['angry']['random'])
        self.assertNotEqual(result['collections']['happy']['agreed'], result['collections']['sad']['agreed'])
        for e in vanilla.EMOTIONS:
            for regime, train in result['collections'][e].items():
                self.assertEqual(len(train), 25)
                self.assertFalse(set(train) & set(result['test_ids']))
                if regime == 'agreed':
                    self.assertTrue(set(train) <= agreed[e])
        with self.assertRaisesRegex(ValueError, 'recognized fitting'):
            vanilla.select(pool, dict(all_agreed, happy=[]), 20, 25)

    def test_manifest_hash_is_portable_and_plans_use_same_prompt_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, plans = self.prepare(root/'a', '/machine-one')
            second, _ = self.prepare(root/'b', '/machine-two')
            self.assertEqual(vanilla.fingerprint(first), vanilla.fingerprint(second))
            for e in vanilla.EMOTIONS:
                for regime, plan in plans[e].items():
                    self.assertEqual(plan['prompt'], vanilla.PROMPT)
                    self.assertEqual(plan['collection_prompt'], vanilla.PROMPT)
                    self.assertEqual(plan['test_ids'], first['test_ids'])
                    self.assertFalse(set(plan['collection_ids']) & set(plan['test_ids']))
            changed = self.registry()
            changed.labels['q00angry']['emotion'] = 'sad'
            with self.assertRaises(ValueError):
                vanilla.prepare(changed, root/'a', 16, 4, 5)

    def test_deliveries_have_no_steering_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plans = self.prepare(root)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            with patch.object(vanilla.task, 'generate', return_value=('#### 2', {})) as generate:
                vanilla.deliver(plans, ['happy', 'sad'], root/'delivery', None, tokenizer, None)
                self.assertEqual(generate.call_count, 12)  # four questions, neutral once + happy + sad
                vanilla.deliver(plans, ['happy', 'sad'], root/'delivery', None, tokenizer, None)
                self.assertEqual(generate.call_count, 12)
            rows = list(vanilla.delivery_rows(plans, root/'delivery').values())
            self.assertTrue(all(r['strength'] == 0. and r['interventions'] == [] for r in rows))
            rows[0]['prompt'] = 'Another prompt'
            (root/'delivery/results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                vanilla.delivery_rows(plans, root/'delivery')

    def test_collection_reuses_regime_overlap_and_resumes_between_checkpoint_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            seen = []
            def generate(record, *args):
                seen.append((record['question_id'], record['style']))
                return record['style'], {}
            def capture(model, fn, decoder, budget):
                style, encoding = fn()
                value = {'neutral': 1., 'angry': 3., 'happy': 5., 'sad': 0.}[style]
                return style, encoding, torch.full((32, 2, 4, 3), value)
            src = ModuleType('src'); src.__path__ = []
            fast = ModuleType('src.fast_dllm'); fast.__path__ = []
            decoder = ModuleType('src.fast_dllm.diffa_generate')
            original_save = vanilla.task.save_tensor
            interrupted = [False]
            def save(torch_module, state, path):
                if (path.parent.name == 'agreed' and path.parent.parent.name == 'happy' and not interrupted[0]
                        and state['completed_ids'][-1] in plans['happy']['random']['collection_ids']):
                    interrupted[0] = True
                    raise RuntimeError('interrupted between regime saves')
                original_save(torch_module, state, path)
            with patch.dict(sys.modules, {'src': src, 'src.fast_dllm': fast, 'src.fast_dllm.diffa_generate': decoder}), patch.object(
                    vanilla.task, 'generate', side_effect=generate), patch.object(vanilla.task.emotion, 'capture', side_effect=capture):
                with patch.object(vanilla.task, 'save_tensor', side_effect=save):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        vanilla.collect(plans, ['happy', 'sad'], manifest['fitting_order'], root, None, None, None)
                vanilla.collect(plans, ['happy', 'sad'], manifest['fitting_order'], root, None, None, None)
                count = len(seen)
                vanilla.collect(plans, ['happy', 'sad'], manifest['fitting_order'], root, None, None, None)
                self.assertEqual(count, len(seen))
            for e, delta in [('happy', 4.), ('sad', -1.)]:
                for regime in vanilla.REGIMES:
                    vector, info = vanilla.task.winners.read_vector(root/e/regime/'style-vectors.pt', e)
                    torch.testing.assert_close(vector, torch.full((3,), delta))
                    self.assertEqual(info['pairs'], 5)
                    self.assertEqual(info['extraction_ids'], plans[e][regime]['collection_ids'])


if __name__ == '__main__':
    unittest.main()
