import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import run_bbh_five_mean as run
from wait_for_experiments import active_jobs


class FiveObjectMeanTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'; source.mkdir(); benchmark = root/'old'; benchmark.mkdir()
        example = dict(input='Five birds.\nOptions:\n'+'\n'.join(f'({c}) Bird {c}' for c in 'ABCDE'), target='(E)')
        path = source/'logical_deduction_five_objects-source.json'
        path.write_text(json.dumps(dict(examples=[example]*250)))
        questions = run.reasoning.bbh_questions(path.read_bytes())
        bank = {q['question_id']: q for q in questions}
        records = {(q, s): dict(question_id=q, style=s, voice='Jenny', audio_file=f'{q}-{s}.wav',
                               audio_sha256=f'{q}-{s}', answer='E') for q in bank for s in ('neutral', 'angry')}
        previous = dict(task='bbh5', bank=bank, recordings=run.reasoning.tasks_for(records, bank),
            source_sha256=run.reasoning.digest(path), prompt=run.reasoning.BBH_PROMPT,
            collection_prompt='OLD THREE OBJECT PROMPT', vector='old-three.pt', vector_sha256='old',
            seed=42, decoder='fast_dllm', max_new_tokens=512)
        (benchmark/'run_config.json').write_text(json.dumps(previous))
        return source, benchmark, records, previous

    def test_disjoint_seeded_split_uses_five_object_prompt_and_own_vector(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, benchmark, records, _ = self.fixture(root)
            with patch.object(run.reasoning, 'source_records', return_value=records):
                plan = run.prepare(source, benchmark, root/'new')
                self.assertEqual(plan, run.prepare(source, benchmark, root/'new'))
            train = {r['question_id'] for r in plan['extraction']}; test = set(plan['bank'])
            self.assertEqual((len(train), len(test), len(train & test)), (125, 125, 0))
            self.assertEqual(len(plan['recordings']), 375)
            self.assertEqual(plan['collection_prompt'], run.reasoning.BBH_PROMPT)
            self.assertEqual(plan['vector'], str((root/'new/style-vectors.pt').absolute()))
            self.assertNotIn('vector_sha256', plan)

    def test_reuses_only_heldout_baselines_never_previous_transfer_intervention(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, benchmark, records, previous = self.fixture(root)
            output = root/'new'; output.mkdir()
            with patch.object(run.reasoning, 'source_records', return_value=records):
                plan = run.prepare(source, benchmark, output)
            rows = [dict(r, reference_answer='E', response='The answer is (E).',
                         prompt=previous['prompt'], seed=42, max_new_tokens=512, decoder='fast_dllm')
                    for r in previous['recordings']]
            (benchmark/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            run.reuse_answers(benchmark, output, plan); run.reuse_answers(benchmark, output, plan)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual(len(saved), 250)
            self.assertEqual({r['condition'] for r in saved}, {'neutral', 'angry'})
            self.assertEqual({r['question_id'] for r in saved}, set(plan['bank']))

    def test_queue_wait_detects_parent_and_stage_worker_not_new_followup_or_tail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'repo'; root.mkdir(); proc = Path(d)/'proc'; proc.mkdir()
            for pid, args in [(801, ['python', 'run_reasoning_transfer.py']),
                              (802, ['python', 'run_reasoning_transfer.py', '--stage', 'gsm8k']),
                              (803, ['python', 'run_bbh_five_mean.py']),
                              (804, ['tail', '-f', 'reasoning-transfer.log'])]:
                entry = proc/str(pid); entry.mkdir(); (entry/'cwd').symlink_to(root)
                (entry/'stat').write_text(f'{pid} (worker) S 0 0')
                (entry/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args))
            self.assertEqual(active_jobs('reasoning-transfer', root, proc), [801, 802])


if __name__ == '__main__':
    unittest.main()
