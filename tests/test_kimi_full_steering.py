from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import kimi_bbh_full_steering as full
import kimi_bbh_layer_diagnostic as diagnostic


class FullSteeringTests(unittest.TestCase):
    def fixture(self, root):
        from tests.test_kimi_bbh_followups import KimiFollowupTests
        original_args, source, rows = KimiFollowupTests().fixture(root)
        benchmark = root/'benchmark'
        for row in rows:
            row.update(prompt=full.OLD_PROMPT, model='model', revision='revision')
        (benchmark/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        previous = json.loads((benchmark/'run_config.json').read_text())
        previous['prompt'] = full.OLD_PROMPT
        (benchmark/'run_config.json').write_text(json.dumps(previous))
        pairs = diagnostic.matched_pairs(rows)
        source.update(prompt=full.OLD_PROMPT, source_config_sha256=diagnostic.digest(benchmark/'run_config.json'),
                      pairs=[pairs[0]], collection_pairs=[pairs[1]])
        (original_args.recovery/'run_config.json').write_text(json.dumps(source))
        target = full.TARGETS[0]
        cached = dict(pairs[0]['neutral'], condition=target['condition'],
                      interventions=[target], response='The answer is (A).')
        (original_args.recovery/'results.jsonl').write_text(json.dumps(cached)+'\n')
        regression_plan = dict(prompt=full.OLD_PROMPT, generation=diagnostic.PARAMS,
            instruct_model='model', instruct_revision='revision',
            vector_source=str(original_args.recovery/'style-vectors.pt'))
        (original_args.output/'run_config.json').write_text(json.dumps(regression_plan))
        regrow = dict(pairs[1]['neutral'], condition=target['condition'], interventions=[target])
        (original_args.output/'results.jsonl').write_text(json.dumps(regrow)+'\n')
        output = root/'full'; output.mkdir()
        args = SimpleNamespace(recovery=original_args.recovery, regressions=original_args.output, output=output)
        return args

    def test_full_plan_has_all_questions_fixed_layers_old_prompt_and_overlap_labels(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            plan = full.prepare(args)
        self.assertEqual(len(plan['pairs']), 4)
        self.assertEqual(plan['prompt'], full.OLD_PROMPT)
        self.assertEqual([(t['layer'], t['group'], t['strength']) for t in plan['targets']],
                         [(16, 'answer', 1.), (17, 'answer', 1.), (18, 'answer', 1.)])
        self.assertEqual([p['question_id'] for p in plan['pairs'] if p['in_vector_collection']], ['q1'])
        self.assertEqual({p['question_id'] for p in plan['pairs']}, {'q0', 'q1', 'q2', 'q3'})

    def test_full_plan_rejects_explicit_reasoning_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            path = args.recovery/'run_config.json'; source = json.loads(path.read_text())
            source['prompt'] = source['prompt'].replace("Let's think step by step.", 'Give concise step-by-step reasoning.')
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'OLD'):
                full.prepare(args)

    def test_reuses_compatible_recovery_and_regression_answers_once_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args = self.fixture(Path(d)); plan = full.prepare(args)
            sources = [args.recovery, args.regressions]
            before = [(p/'results.jsonl').read_bytes() for p in sources]
            full.reuse_answers(plan, args.output, sources)
            full.reuse_answers(plan, args.output, sources)
            rows = diagnostic.read_rows(args.output/'results.jsonl')
            self.assertEqual(len(rows), 2)
            self.assertEqual({r['question_id'] for r in rows}, {'q0', 'q1'})
            self.assertEqual(before, [(p/'results.jsonl').read_bytes() for p in sources])

    def test_reuse_rejects_changed_prompt_audio_or_steering_strength(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); plan = full.prepare(args)
            task = plan['pairs'][0]; target = plan['targets'][0]
            row = diagnostic.read_rows(args.recovery/'results.jsonl')[0]
            self.assertTrue(full.reusable(row, task, target, plan))
            for field, changed in [('prompt', 'new prompt'), ('audio_sha256', 'different audio'), ('revision', 'different checkpoint')]:
                self.assertFalse(full.reusable(dict(row, **{field: changed}), task, target, plan))
            self.assertFalse(full.reusable(dict(row, interventions=[dict(target, strength=2.)]), task, target, plan))

    def test_cpu_summary_separates_collection_seen_and_unseen_and_retains_sources(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args = self.fixture(Path(d)); plan = full.prepare(args)
            rows = []
            for pair in plan['pairs']:
                for style in ('neutral', 'angry'):
                    rows.append(dict(pair[style], in_vector_collection=pair['in_vector_collection']))
                for target in plan['targets']:
                    rows.append(dict(pair['neutral'], condition=target['condition'], response='The answer is (A).',
                                     in_vector_collection=pair['in_vector_collection']))
            path = args.output/'results.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            before = path.read_bytes()
            full.report(args.output)
            summary = json.loads((args.output/'summary.json').read_text())
            self.assertEqual(path.read_bytes(), before)
            sizes = {(r['subset'], r['condition']): r['n'] for r in summary['accuracy']}
            condition = plan['targets'][0]['condition']
            self.assertEqual([sizes[s, condition] for s in ('all', 'collection-unseen', 'collection-seen')], [4, 3, 1])
            self.assertEqual(summary['prompt'], full.OLD_PROMPT)


if __name__ == '__main__':
    unittest.main()
