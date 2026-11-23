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

    def test_agreement_reports_all_clips_and_exact_completed_pairs_without_neutral_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plans = self.prepare(root)
            rows = [dict(question_id=q, condition=s) for q, s in
                    [('q00', 'neutral'), ('q02', 'neutral'), ('q00', 'happy'), ('q01', 'happy')]]
            stats = vanilla.report_agreement(plans, root, ['happy', 'sad'], rows)
            self.assertEqual(stats['happy']['all_clips']['classified'], 16)
            self.assertEqual(stats['happy']['all_clips']['agreed'], 15)
            self.assertEqual(stats['happy']['matched_accuracy_pairs']['questions'], 1)
            self.assertEqual(stats['happy']['matched_accuracy_pairs']['agreement_pct'], 100.)
            self.assertEqual(stats['happy']['agreed_fitting']['agreed'], 5)
            self.assertIsNone(stats['sad']['matched_accuracy_pairs']['agreement_pct'])
            self.assertEqual(vanilla.agreement_counts({}, ['q00'], 'sad')['missing'], 1)

    def test_agreed_accuracy_uses_common_recognized_questions_and_only_completed_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self.prepare(Path(directory))
            rows = [dict(question_id=q, condition=s, response='wrong', metrics=dict(correct=False)) for q, s in
                    [('q00', 'neutral'), ('q00', 'happy'), ('q00', 'sad'), ('q03', 'neutral'), ('q03', 'happy')]]
            eligible, complete = vanilla.agreed_cohort(manifest, rows, ['happy', 'sad'])
            self.assertIn('q00', eligible)  # angry is misclassified, but isn't required here
            self.assertNotIn('q01', eligible)  # happy is misclassified
            self.assertNotIn('q02', eligible)  # sad is misclassified
            self.assertEqual(complete, ['q00'])  # q03 has no sad response yet
            _, all_three = vanilla.agreed_cohort(manifest, rows, ['happy', 'sad'], ['angry', 'happy', 'sad'])
            self.assertEqual(all_three, [])
            with self.assertRaises(ValueError):
                vanilla.agreed_cohort(manifest, rows, ['happy', 'sad'], ['angry'])

    def test_filtered_scoring_keeps_same_baseline_and_preserves_unfiltered_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            scoped = vanilla.prepare_deliveries(self.registry(), manifest, plans, root)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            def generate(record, *args):
                return ('#### 3' if record['style'] == 'sad' else '#### 2'), {}
            with patch.object(vanilla.task, 'generate', side_effect=generate):
                vanilla.deliver(scoped, ['happy', 'sad'], root/'delivery', None, tokenizer, None)
            untouched = root/'happy/delivery-report'
            untouched.mkdir(exist_ok=True)
            (untouched/'summary.json').write_text('original unfiltered report')
            def score(plan, rows, folder):
                return [dict(r, metrics={'flexible-numeric': r['response'] == '#### 2'}) for r in rows], {}
            with patch.object(vanilla.task.winners, 'score_rows', side_effect=score):
                vanilla.report_agreed_accuracy(plans, root, ['happy', 'sad'], tokenizer, agreement_emotions=['happy', 'sad'])
            reports = root/'agreed-accuracy/happy-sad'
            for e in ('happy', 'sad'):
                summary = json.loads((reports/e/'summary.json').read_text())
                self.assertEqual(summary['matched_questions'], 14)
                self.assertEqual(summary['paired'][0]['baseline_correct'], 14)
                self.assertEqual(summary['paired'][0]['steered_correct'], 14 if e == 'happy' else 0)
            compared = json.loads((reports/'emotion-comparisons.json').read_text())
            self.assertEqual(compared[0]['pairs'], 14)
            self.assertEqual((untouched/'summary.json').read_text(), 'original unfiltered report')
            with patch.object(vanilla.task.winners, 'score_rows', side_effect=score):
                vanilla.report_agreed_accuracy(plans, root, ['happy', 'sad'], tokenizer)
            # Default reports MUST NOT require the other emotion to be recognized.
            for e in ('happy', 'sad'):
                summary = json.loads((root/'agreed-accuracy'/e/e/'summary.json').read_text())
                self.assertEqual(summary['matched_questions'], 15)
                self.assertEqual(summary['required_agreement'], [e])
                self.assertEqual(summary['paired'][0]['baseline_correct'], 15)
                self.assertEqual(summary['paired'][0]['steered_correct'], 15 if e == 'happy' else 0)

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

    def test_extending_finished_deliveries_keeps_existing_answers_and_frozen_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            frozen = {p: p.read_bytes() for p in root.glob('*/*/run_config.json')}
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            with patch.object(vanilla.task, 'generate', return_value=('#### 2', {})) as generate:
                vanilla.deliver(plans, ['angry'], root/'delivery', None, tokenizer, None)
                original = (root/'delivery/results.jsonl').read_bytes()
                self.assertEqual(generate.call_count, 8)
                expanded = vanilla.prepare_deliveries(self.registry(), manifest, plans, root)
                vanilla.deliver(expanded, ['angry'], root/'delivery', None, tokenizer, None)
                self.assertEqual(generate.call_count, 32)  # only 12 remaining questions x two deliveries
                self.assertTrue((root/'delivery/results.jsonl').read_bytes().startswith(original))
                vanilla.deliver(expanded, ['angry'], root/'delivery', None, tokenizer, None)
                self.assertEqual(generate.call_count, 32)
            for p, data in frozen.items():
                self.assertEqual(p.read_bytes(), data)
            self.assertEqual(plans['angry']['random']['test_ids'], manifest['test_ids'])
            self.assertEqual(len(vanilla.delivery_scope(plans, root)['angry']['random']['test_ids']), 16)

    def test_fitting_answers_fill_full_deliveries_and_are_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            expanded = vanilla.prepare_deliveries(self.registry(), manifest, plans, root)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            q = plans['angry']['random']['collection_ids'][0]
            row = next(r for r in expanded['angry']['random']['recordings'] if r['question_id'] == q and r['condition'] == 'angry')
            saved = dict(row, **vanilla.RECIPE, prompt=vanilla.PROMPT, response='already saved', strength=0., interventions=[])
            with patch('diffa_collection_report.collection_rows', return_value=[saved]):
                vanilla.reuse_collection_deliveries(plans, expanded, ['angry'], root, tokenizer)
                vanilla.reuse_collection_deliveries(plans, expanded, ['angry'], root, tokenizer)
            with patch.object(vanilla.task, 'generate', return_value=('new', {})) as generate:
                vanilla.deliver(expanded, ['angry'], root/'delivery', None, tokenizer, None)
                self.assertEqual(generate.call_count, 31)
            self.assertEqual(vanilla.delivery_rows(expanded, root/'delivery')[q, 'angry']['response'], 'already saved')
            with patch('diffa_collection_report.collection_rows', return_value=[dict(saved, prompt='wrong prompt')]):
                with self.assertRaisesRegex(ValueError, 'cannot be reused'):
                    vanilla.reuse_collection_deliveries(plans, expanded, ['angry'], root, tokenizer)

    def test_both_vectors_steer_all_unfitted_questions_and_reuse_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            expanded = vanilla.prepare_deliveries(self.registry(), manifest, plans, root)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            with patch.object(vanilla.task, 'generate', return_value=('#### 2', {})):
                vanilla.deliver(expanded, ['angry'], root/'delivery', None, tokenizer, None)
            seen = []
            def descriptor(plan, folder):
                return dict(plan['target'], emotion=plan['emotion'], vector_sha256='vector-'+plan['regime'],
                            vector=str(folder/'style-vectors.pt'), extraction_ids=plan['collection_ids'])
            def generate_job(job, folder, *args):
                self.assertEqual(job['strengths'], [0., 1.])
                saved = vanilla.steering_rows(job, folder)
                self.assertEqual(sum(r['strength'] == 0. for r in saved.values()), 11)
                with (folder/'results.jsonl').open('a') as stream:
                    for record in job['target']['records']:
                        key = record['question_id'], record['voice'], 'L31-answer-a1'
                        if key in saved:
                            continue
                        seen.append((job['regime'], record['question_id']))
                        stream.write(json.dumps(dict(record, response='#### 2', **vanilla.RECIPE,
                            prompt=vanilla.PROMPT, condition='L31-answer-a1', strength=1.,
                            vector_sha256=job['source']['vector_sha256'],
                            interventions=[dict(layer=31, group='answer', strength=1.)]))+'\n')
            with patch.object(vanilla.task, 'descriptor', side_effect=descriptor), patch.object(
                    vanilla.task.winners, 'generate_job', side_effect=generate_job), patch.object(vanilla.task, 'report_steering'):
                vanilla.steer(plans, expanded, ['angry'], root, None, tokenizer, None)
                vanilla.steer(plans, expanded, ['angry'], root, None, tokenizer, None)
            self.assertEqual(len(seen), 22)
            for regime in vanilla.REGIMES:
                expected = [r['question_id'] for r in expanded['angry'][regime]['target']['records']
                            if r['question_id'] not in plans['angry'][regime]['collection_ids']]
                self.assertEqual([q for r, q in seen if r == regime], expected)
                self.assertFalse(set(expected) & set(plans['angry'][regime]['collection_ids']))
            folder = root/'angry/random/strength-1-unfitted'
            job = json.loads((folder/'run_config.json').read_text())
            rows = list(vanilla.steering_rows(job, folder).values())
            rows[-1]['vector_sha256'] = 'changed-vector'
            (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'Steering cache differs'):
                vanilla.steering_rows(job, folder)

    def test_single_pass_generates_all_deliveries_and_collects_without_repeated_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, plans = self.prepare(root)
            scoped = vanilla.prepare_deliveries(self.registry(), manifest, plans, root)
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            seen = []
            def generate(record, *args):
                seen.append((record['question_id'], record['style']))
                return record['style'], {}
            def capture(model, fn, decoder, budget):
                style, encoding = fn()
                value = {'neutral': 1., 'happy': 5., 'sad': 0.}[style]
                return style, encoding, torch.full((32, 2, 4, 3), value)
            src = ModuleType('src'); src.__path__ = []
            fast = ModuleType('src.fast_dllm'); fast.__path__ = []
            decoder = ModuleType('src.fast_dllm.diffa_generate')
            with patch.dict(sys.modules, {'src': src, 'src.fast_dllm': fast, 'src.fast_dllm.diffa_generate': decoder}), patch.object(
                    vanilla.task, 'generate', side_effect=generate), patch.object(vanilla.task.emotion, 'capture', side_effect=capture):
                order = manifest['test_ids']+manifest['fitting_order']
                vanilla.generate_and_collect(plans, scoped, ['happy', 'sad'], order, root, None, tokenizer, None)
                self.assertEqual(len(seen), 48)
                self.assertEqual(len(set(seen)), 48)
                vanilla.generate_and_collect(plans, scoped, ['happy', 'sad'], order, root, None, tokenizer, None)
                self.assertEqual(len(seen), 48)
            self.assertEqual(len(vanilla.delivery_rows(scoped, root/'delivery')), 48)
            for e, delta in [('happy', 4.), ('sad', -1.)]:
                for regime in vanilla.REGIMES:
                    vector, info = vanilla.task.winners.read_vector(root/e/regime/'style-vectors.pt', e)
                    torch.testing.assert_close(vector, torch.full((3,), delta))
                    self.assertEqual(info['pairs'], 5)


if __name__ == '__main__':
    unittest.main()
