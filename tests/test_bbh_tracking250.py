import json
from pathlib import Path
import tempfile
import unittest

import run_bbh_tracking250 as run


class TrackingFollowupTests(unittest.TestCase):
    def plan(self, root):
        bank = {f'q{i}': dict(original_input='Swap books.', original_target='(A)') for i in range(250)}
        rows = [dict(question_id=q, voice='Jenny', style=s, condition=s, answer='(A)',
                     audio_file=str(root/f'{q}-{s}.wav'), audio_sha256=f'{q}-{s}')
                for q in bank for s in run.delivery.STYLES]
        return dict(bank=bank, recordings=rows, prompt=run.delivery.PROMPTS[run.TASK],
                    harness_task='bbh_cot_zeroshot_'+run.TASK)

    def test_fixed_disjoint_split_preserves_tracking_prompt_and_ignores_happy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = self.plan(root)
            (root/'plan.jsonl').write_text(''.join(json.dumps(dict(r, audio_path=Path(r['audio_file']).name))+'\n'
                                                  for r in plan['recordings']))
            split = run.split_plan(plan, root)
            train = {r['question_id'] for r in split['extraction']}
            test = {r['question_id'] for r in split['test']}
            self.assertEqual((len(train), len(test), len(train & test)), (125, 125, 0))
            self.assertEqual(split['test'][0]['question_id'], 'q125')
            self.assertEqual(split['prompt'], run.delivery.PROMPTS[run.TASK])
            self.assertEqual(split['harness_task'], 'bbh_cot_zeroshot_'+run.TASK)
            self.assertNotIn('happy', split['test'][0])
            self.assertEqual((split['layer'], split['group'], split['strength']), (31, 'answer', 1))

    def test_reuse_only_matching_held_out_neutral_angry_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'previous'; source.mkdir(); output = root/'new'; output.mkdir()
            plan = self.plan(root)
            rows = [dict(r, reference_answer=r['answer'], response='The answer is (A).',
                         prompt=plan['prompt'], seed=42, max_new_tokens=512, decoder='fast_dllm')
                    for r in plan['recordings']]
            (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            allowed = {f'q{i}' for i in range(125, 250)}
            run.copy_answers(source, output, plan, allowed)
            run.copy_answers(source, output, plan, allowed)
            saved = run.read_rows(output/'results.jsonl')
            self.assertEqual(len(saved), 250)
            self.assertEqual({r['style'] for r in saved}, {'neutral', 'angry'})
            self.assertEqual({r['question_id'] for r in saved}, allowed)
            rows[0]['audio_sha256'] = 'different-waveform'
            (source/'results.jsonl').write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'recipe'):
                run.copy_answers(source, output, plan)


if __name__ == '__main__':
    unittest.main()
