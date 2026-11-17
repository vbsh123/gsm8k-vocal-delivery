import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import phi4_answer_only as run


class AnswerOnlyTests(unittest.TestCase):
    def fixture(self, root, task='gsm8k'):
        source = root/'source'; source.mkdir(); output = root/'followup'
        prompt = run.gsm.ORIGINAL if task == 'gsm8k' else run.bbh.PROMPT
        records = [dict(question_id=f'q{i}', voice='Jenny', style=s, condition=s,
            audio_sha256=f'{i}-{s}', audio_file=f'/audio/{i}-{s}.wav', reference_answer='2')
            for i in range(3) for s in ('neutral','angry')]
        bank = {f'q{i}':dict(question='One plus one?',answer='2') for i in range(3)}
        plan = dict(recipe=run.MODELS['phi4'], recordings=records, bank=bank, prompt=prompt,
            prompt_variant='original', seed=42, max_new_tokens=512, interventions=[], scoring='Upstream GSM8K filters.')
        (source/'run_config.json').write_text(json.dumps(plan))
        # One complete correct pair, one complete incorrect pair, one unfinished pair.
        rows = [dict(t, response='#### 2' if t['question_id']=='q0' else '#### 3',
            response_tokens=5, seed=42, max_new_tokens=512, interventions=[], prompt=prompt,
            model=run.MODELS['phi4']['model'],revision=run.MODELS['phi4']['revision']) for t in records[:5]]
        data = ''.join(json.dumps(r)+'\n' for r in rows)
        (source/'results.jsonl').write_text(data+'{"question_id":')
        return source, output, plan, rows

    def test_freezes_all_completed_pairs_without_outcome_filter_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            source, output, old, rows = self.fixture(Path(d))
            before = (source/'results.jsonl').read_bytes()
            dest, plan = run.prepare_task('gsm8k', source, output)
            self.assertEqual(len(plan['recordings']), 4)
            self.assertEqual(set(plan['bank']), {'q0','q1'})
            self.assertEqual(plan['prompt'], old['prompt'].replace("Let's think step by step.", run.DIRECTIVE))
            self.assertIn('#### <number>', plan['prompt'])
            self.assertEqual(plan['seed'], 42)
            self.assertEqual(plan['max_new_tokens'], 512)
            self.assertEqual(run.read_rows(output/'original/results.jsonl'), rows[:4])
            self.assertEqual(before, (source/'results.jsonl').read_bytes())
            selection = json.loads((output/'selection.json').read_text())
            self.assertTrue(selection['incomplete_final_line_omitted'])
            self.assertEqual(selection['excluded_unpaired_answers'], 1)
            self.assertFalse((dest/'results.jsonl').exists())

    def test_resume_does_not_extend_cohort_when_source_grows(self):
        with tempfile.TemporaryDirectory() as d:
            source, output, old, rows = self.fixture(Path(d))
            _, frozen = run.prepare_task('gsm8k', source, output)
            extra = dict(rows[-1], style='angry', condition='angry', audio_sha256='2-angry')
            (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows+[extra]))
            _, resumed = run.prepare_task('gsm8k', source, output)
            self.assertEqual(frozen, resumed)
            self.assertEqual(len(resumed['recordings']), 4)

    def test_bbh_keeps_final_answer_format_and_does_not_keep_cot_request(self):
        with tempfile.TemporaryDirectory() as d:
            source, output, old, rows = self.fixture(Path(d), 'bbh3')
            _, plan = run.prepare_task('bbh3', source, output)
            self.assertIn('The answer is (<letter>).', plan['prompt'])
            self.assertIn(run.DIRECTIVE, plan['prompt'])
            self.assertNotIn("Let's think step by step.", plan['prompt'])

    def test_rejects_changed_seed_or_recording(self):
        for field, value in [('seed', 7), ('audio_sha256', 'changed')]:
            with tempfile.TemporaryDirectory() as d:
                source, output, old, rows = self.fixture(Path(d))
                rows[0][field] = value
                (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError): run.prepare_task('gsm8k', source, output)

    def test_process_selection_only_targets_phi4_gsm_and_its_children_in_this_repo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); proc = root/'proc'; proc.mkdir(); repo=root/'repo'; repo.mkdir()
            def process(pid, parent, args, cwd=repo):
                folder=proc/str(pid); folder.mkdir(); (folder/'cwd').symlink_to(cwd)
                (folder/'cmdline').write_bytes(b'\0'.join(s.encode() for s in args)+b'\0')
                (folder/'stat').write_text(f'{pid} (process) '+' '.join(['S',str(parent)]+['0']*17+['123']))
            process(101,1,['bash','run_gsm_candidate.sh','phi4'])
            process(102,101,['python','evaluate_gsm_candidates.py','--model','phi4','--mode','evaluate'])
            process(103,101,['flock','some.lock','true'])
            process(104,1,['python','evaluate_bbh_candidates.py','--model','phi4','--mode','evaluate'])
            process(105,1,['python','evaluate_gsm_candidates.py','--model','minicpm','--mode','evaluate'])
            process(106,1,['python','evaluate_gsm_candidates.py','--model','phi4','--mode','summary'])
            self.assertEqual(set(run.targeted_jobs(repo,proc)),{101,102,103})

    def test_reports_same_four_cell_cohort_and_interaction_without_regenerating(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source, _, _, rows=self.fixture(root)
            run.prepare_task('gsm8k',source,root/'gsm8k')
            plan=json.loads((root/'gsm8k/answer-only/run_config.json').read_text())
            # Only q0 complete in new prompt, so interaction must use q0 only.
            new=[dict(r,prompt=plan['prompt'],response='#### 2') for r in rows[:2]]
            (root/'gsm8k/answer-only/results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
            run.report(SimpleNamespace(output=root))
            report=json.loads((root/'gsm8k/comparison.json').read_text())
            self.assertEqual(report['pairs'],1)
            self.assertEqual(report['emotion_by_prompt_interaction']['gain_difference_pp'],0.)


if __name__ == '__main__': unittest.main()
