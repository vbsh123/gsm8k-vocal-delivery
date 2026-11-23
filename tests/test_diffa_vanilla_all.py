import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_vanilla_all as all_tasks


class AllVanillaTests(unittest.TestCase):
    def registry(self, root, name, total=16):
        root.mkdir(parents=True, exist_ok=True)
        options, labels, examples = {}, {}, []
        ids = [f'{name}_{i:02}' for i in range(total)]
        for i, q in enumerate(ids):
            answer = '' if name == 'ifeval' else 'A' if name.startswith('bbh') else '2'
            example = dict(question_id=q, question='Question '+q, answer=answer)
            if name == 'ifeval':
                example.update(key=i, prompt=example['question'], instruction_id_list=['keywords:existence'],
                               kwargs=[dict(keywords=['orange'])])
            elif name.startswith('bbh'):
                example.update(original_input='Original '+q, original_target='(A)')
            examples.append(example)
            for style in all_tasks.STYLES:
                record = dict(question_id=q, style=style, voice='en-US-JennyNeural', seconds=1.,
                              audio_file=str(root/(q+style+'.wav')), audio_sha256=q+style, reference_answer=answer)
                options[name, q, style] = [dict(record=record, question=example['question'], answer=answer)]
                labels[q+style] = dict(classification_status='classified', emotion=style if style != 'neutral' else 'sad')
        (root/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
        def choose(dataset, q, style, **kw):
            self.assertEqual(dataset, name)
            item = options[dataset, q, style][0]
            for key in ('question', 'answer'):
                if key in kw:
                    self.assertEqual(kw[key], item[key])
            if 'expected' in kw:
                self.assertEqual(kw['expected'], item['record']['audio_sha256'])
            return item
        return SimpleNamespace(neutral_ids=lambda dataset: ids, choose=choose, options=options, labels=labels)

    def test_task_prompts_are_vanilla_and_ifeval_keeps_approved_prompt(self):
        for total, count in all_tasks.SPECS.values():
            self.assertEqual(count, total//2)
        self.assertEqual(all_tasks.prompt_for('ifeval'), 'Follow the instructions in the audio.')
        for name in all_tasks.DATASETS:
            prompt = all_tasks.prompt_for(name)
            self.assertNotIn('step by step', prompt)
            self.assertNotIn('reasoning', prompt)
            if name.startswith('bbh'):
                self.assertIn('The answer is (<letter>).', prompt)
                self.assertNotIn('####', prompt)
            elif name != 'ifeval':
                self.assertIn('#### <number>', prompt)

    def test_random_is_unfiltered_and_agreed_uses_available_up_to_cap(self):
        pool = [f'q{i:02}' for i in range(20)]
        eligible = dict(angry=pool[:15], happy=pool[5:], sad=pool[8:11])
        order, selected = all_tasks.selections(pool, eligible, 5, 'bbh5')
        self.assertEqual(set(order), set(pool))
        for e in all_tasks.engine.EMOTIONS:
            self.assertEqual(len(selected[e]['random']), 5)
            self.assertEqual(len(selected[e]['agreed']), 3 if e == 'sad' else 5)
            self.assertTrue(set(selected[e]['agreed']) <= set(eligible[e]))
            self.assertEqual(selected[e]['random'], selected['angry']['random'])
        _, different = all_tasks.selections(pool, eligible, 5, 'svamp')
        self.assertEqual([len(different[e]['agreed']) for e in all_tasks.engine.EMOTIONS], [5, 5, 3])
        with self.assertRaisesRegex(ValueError, 'recognized'):
            all_tasks.selections(pool, dict(eligible, sad=[]), 5, 'svamp')

    def test_all_tasks_generate_collect_in_one_pass_with_correct_prompt_and_budget(self):
        for name in all_tasks.NEXT:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry = self.registry(root/'audio', name)
                output = root/'output'
                manifest, plans = all_tasks.prepare(name, registry, output, spec=(16, 5))
                self.assertNotIn('test_ids', manifest)  # no reserved natural evaluation subset
                scoped = all_tasks.engine.prepare_deliveries(registry, manifest, plans, output)
                self.assertEqual(len(scoped['angry']['random']['test_ids']), 16)
                seen = []
                budget = 1024 if name == 'ifeval' else 512
                def generate(record, plan, prompt, *args):
                    self.assertEqual(prompt, all_tasks.prompt_for(name))
                    self.assertEqual(plan['max_new_tokens'], budget)
                    seen.append((record['question_id'], record['style']))
                    return record['style'], {}
                def capture(model, fn, decoder, max_tokens):
                    self.assertEqual(max_tokens, budget)
                    style, encoding = fn()
                    return style, encoding, torch.full((32, 2, 4, 3), 1. if style == 'neutral' else 3.)
                src = ModuleType('src'); src.__path__ = []
                fast = ModuleType('src.fast_dllm'); fast.__path__ = []
                decoder = ModuleType('src.fast_dllm.diffa_generate')
                tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
                with patch.dict(sys.modules, {'src': src, 'src.fast_dllm': fast, 'src.fast_dllm.diffa_generate': decoder}), patch.object(
                        all_tasks.engine.task, 'generate', side_effect=generate), patch.object(all_tasks.engine.task.emotion, 'capture', side_effect=capture):
                    all_tasks.engine.generate_and_collect(plans, scoped, ['angry', 'happy', 'sad'], manifest['delivery_ids'],
                                                         output, None, tokenizer, None)
                    self.assertEqual(len(seen), 64)
                    self.assertEqual(len(set(seen)), 64)
                    all_tasks.engine.generate_and_collect(plans, scoped, ['angry', 'happy', 'sad'], manifest['delivery_ids'],
                                                         output, None, tokenizer, None)
                    self.assertEqual(len(seen), 64)
                for e in all_tasks.engine.EMOTIONS:
                    for regime in all_tasks.engine.REGIMES:
                        plan = plans[e][regime]
                        job = all_tasks.engine.steering_job(plan, output/e/regime, scoped[e][regime])
                        target_ids = {r['question_id'] for r in job['target']['records']}
                        self.assertEqual(len(target_ids), 11)
                        self.assertFalse(target_ids & set(plan['collection_ids']))
                        self.assertEqual(job['target']['max_new_tokens'], budget)
                        primary = 'strict_prompt' if name == 'ifeval' else 'flexible-extract' if name.startswith('bbh') else 'flexible-numeric'
                        self.assertEqual(job['target']['primary_metric'], primary)
                        if name.startswith('bbh'):
                            self.assertTrue(all('original_input' in row for row in job['target']['bank'].values()))
                        if name == 'ifeval':
                            self.assertTrue(all('instruction_id_list' in row for row in job['target']['bank'].values()))

    def test_missing_classifier_label_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry(root/'audio', 'svamp')
            registry.labels.pop('svamp_00sad')
            with self.assertRaisesRegex(ValueError, 'Missing saved emotion2vec'):
                all_tasks.prepare('svamp', registry, root/'output', spec=(16, 5))


if __name__ == '__main__':
    unittest.main()
