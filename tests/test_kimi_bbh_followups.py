from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import analyze_kimi_reasoning_prompts as analysis
import kimi_bbh_layer_diagnostic as diagnostic
import kimi_bbh_regressions as regression


class KimiFollowupTests(unittest.TestCase):
    def fixture(self, root):
        recovery = root/'recoveries'; recovery.mkdir()
        benchmark = root/'benchmark'; benchmark.mkdir()
        output = root/'regressions'; output.mkdir()
        bank = {f'q{i}': dict(original_input='Which?\nOptions:\n(A) Red\n(B) Blue\n(C) Green',
                              original_target='(A)', answer='A') for i in range(4)}
        pairs, rows = [], []
        for i in range(4):
            pair = dict(question_id=f'q{i}', voice='v')
            for style in ('neutral', 'angry'):
                correct = (style == 'angry') if i == 0 else ((style == 'neutral') if i in (1, 2) else True)
                audio = root/f'q{i}-{style}.wav'; audio.write_bytes(b'example audio '+bytes([i]))
                row = dict(question_id=f'q{i}', voice='v', style=style, condition=style,
                    prompt='prompt', seed=42, max_new_tokens=512, reference_answer='A',
                    audio_file=str(audio), audio_sha256=diagnostic.digest(audio),
                    response='The answer is (A).' if correct else 'The answer is (B).',
                    harness={'flexible-extract': {'correct': correct}})
                pair[style] = row; rows.append(row)
            pairs.append(pair)
        (benchmark/'run_config.json').write_text(json.dumps(dict(bank=bank, prompt='prompt')))
        (benchmark/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        plan = dict(protocol=diagnostic.PROTOCOL, benchmark=str(benchmark), prompt='prompt', seed=42,
            source_config_sha256=diagnostic.digest(benchmark/'run_config.json'),
            layers=[0], groups=['audio', 'answer'], pairs=[pairs[0]], collection_pairs=[pairs[1]],
            generation=diagnostic.PARAMS, instruct_model='model', instruct_revision='revision', source_revision='source')
        (recovery/'run_config.json').write_text(json.dumps(plan))
        sweep = [dict(pairs[0]['neutral'], condition=diagnostic.condition_name(0, g)) for g in plan['groups']]
        (recovery/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sweep))
        torch.save(dict(protocol=diagnostic.PROTOCOL, question_ids=['q1'], audio_emotion=torch.full((28, 3), 11.), emotion=torch.full((28, 3), 22.)),
                   recovery/'style-vectors.pt')
        args = SimpleNamespace(recovery=recovery, output=output, targets=['audio:0', 'answer:0'],
                               top_per_group=1, wait=False, model_source=Path('external/Kimi-Audio'))
        return args, plan, rows

    def test_regression_selection_is_reverse_of_recoveries(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, rows = self.fixture(Path(d))
            self.assertEqual([p['question_id'] for p in regression.regressions(rows)], ['q1', 'q2'])
            self.assertEqual([p['question_id'] for p in diagnostic.recovery_pairs(rows)], ['q0'])

    def test_layer_selection_requires_complete_sweep_and_ranks_recoveries_only(self):
        plan = dict(pairs=[{'question_id': 'q', 'voice': 'v'}], layers=[0, 1, 2], groups=['audio'])
        rows = [dict(question_id='q', voice='v', condition=diagnostic.condition_name(i, 'audio'), prompt='p', seed=42) for i in range(3)]
        expected, actual = regression.complete_recovery_rows(dict(plan, prompt='p', seed=42), rows[:-1])
        self.assertEqual((len(expected), len(actual)), (3, 2))
        scores = {'flexible-extract': [{'correct': False}, {'correct': True}, {'correct': True}],
                  'strict-match': [{'correct': True}, {'correct': True}, {'correct': True}]}
        chosen = regression.rank_targets(plan, rows, scores, 1)
        self.assertEqual(chosen[0]['layer'], 1)  # Tie broken by fixed layer order, not regression performance.
        with self.assertRaisesRegex(ValueError, 'completed'):
            regression.rank_targets(plan, rows[:-1], {k: v[:-1] for k, v in scores.items()}, 1)

    def test_regression_plan_marks_collection_overlap_and_freezes_choices(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(d))
            plan = regression.prepare(args)
            self.assertEqual(plan['overlap_with_collection'], ['q1'])
            self.assertEqual([p['question_id'] for p in plan['pairs']], ['q1', 'q2'])
            self.assertEqual(regression.prepare(args), plan)
            args.targets = ['audio:1']
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                regression.prepare(args)

    def test_selected_targets_reuse_vectors_and_resume_without_collection(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(d))
            plan = regression.prepare(args)
            calls = []
            def generate(model, row, prompt, layer, group, direction):
                calls.append((row['question_id'], group, float(direction[0]), prompt))
                return 'The answer is (A).'
            with patch.object(diagnostic, 'load_model', return_value=object()) as load, \
                 patch.object(diagnostic, 'generate_steered', side_effect=generate), \
                 patch.object(regression.subprocess, 'check_output', return_value=''):
                # Lock lives under the repository outputs; isolate it in this test.
                lock = Path(d)/'gpu.lock'
                original_open = Path.open
                def open_file(path, *a, **kw):
                    return original_open(lock if str(path) == 'outputs/jenny-task-vectors.lock' else path, *a, **kw)
                with patch.object(Path, 'open', open_file):
                    regression.evaluate(plan, args)
                    regression.evaluate(plan, args)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(len(calls), 4)
            self.assertEqual({(g, v) for _, g, v, _ in calls}, {('audio', 11.), ('answer', 22.)})
            rows = diagnostic.read_rows(args.output/'results.jsonl')
            self.assertEqual(len(rows), 8)  # Four original references plus four steered answers.
            self.assertEqual({r['question_id'] for r in rows}, {'q1', 'q2'})

    def prompt_fixture(self, root):
        old = root/'original'; new = root/'explicit'; old.mkdir(); new.mkdir()
        original_prompt = "Task. Let's think step by step. The answer is (<letter>)."
        bank = {'q': dict(original_input='Which?\nOptions:\n(A) Red\n(B) Blue\n(C) Green', original_target='(A)')}
        for folder, prompt in [(old, original_prompt), (new, original_prompt.replace("Let's think step by step.", analysis.REASONING_SENTENCE))]:
            records, rows = [], []
            for style in ('neutral', 'angry'):
                record = dict(question_id='q', voice='v', style=style, audio_sha256='hash-'+style)
                records.append(record)
                text = 'The answer is (B).' if folder == old and style == 'neutral' else 'The answer is (A).'
                rows.append(dict(record, condition=style, response=text, prompt=prompt, seed=42,
                    max_new_tokens=512, model='kimi', revision='pin', reference_answer='A'))
            plan = dict(bank=bank, prompt=prompt, seed=42, max_new_tokens=512,
                        recipe={'model': 'kimi', 'revision': 'pin'}, recordings=records)
            (folder/'run_config.json').write_text(json.dumps(plan))
            (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return old, new

    def test_prompt_join_checks_identical_audio_and_only_reasoning_sentence_change(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = self.prompt_fixture(Path(d))
            _, common, rows = analysis.aligned_rows(old, new)
            self.assertEqual((len(common), len(rows)), (1, 4))
            path = new/'run_config.json'; plan = json.loads(path.read_text())
            plan['recordings'][0]['audio_sha256'] = 'different'
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'different audio'):
                analysis.aligned_rows(old, new)

    def test_interaction_is_change_in_delivery_gap_not_new_prompt_gain_alone(self):
        # Original angry advantage 100 pp; explicit prompt closes it completely.
        result = analysis.interaction_interval([-1, -1])
        self.assertEqual(result, {'gain_pp': -100., 'ci95_gain_pp': [-100., -100.]})

    def test_cpu_prompt_analysis_preserves_manual_reasoning_annotations(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); old, new = self.prompt_fixture(root); output = root/'analysis'
            raw_before = [(p/'results.jsonl').read_bytes() for p in (old, new)]
            analysis.analyze(old, new, output)
            path = output/'reasoning-review.csv'
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertNotIn('style', rows[0])
            rows[0]['reasoning_status'] = 'none'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            analysis.analyze(old, new, output)
            with path.open(newline='') as f:
                restored = {r['sample_id']: r for r in csv.DictReader(f)}
            self.assertEqual(restored[rows[0]['sample_id']]['reasoning_status'], 'none')
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['matched_questions'], 1)
            self.assertEqual(len(summary['cells']), 4)
            self.assertEqual(raw_before, [(p/'results.jsonl').read_bytes() for p in (old, new)])


if __name__ == '__main__':
    unittest.main()
