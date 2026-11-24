from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phi4_gsm_bare_eval as run


class PhiBareEvalTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'night/run_config.json'; source.parent.mkdir()
        bank = {f'q{i}':dict(question_id=f'q{i}', question=f'What is {i} plus two?', answer=str(i+2)) for i in range(3)}
        records = []
        for q in bank:
            for s in ('neutral','angry','happy','sad'):
                path = root/(q+'-'+s+'.wav'); path.write_bytes((q+s).encode())
                records.append(dict(question_id=q, condition=s, style=s, voice='en-US-JennyNeural',
                    audio_file=str(path), audio_sha256=run.sha(path), reference_answer=bank[q]['answer']))
        parent = dict(task='gsm8k',model='phi',prompt=run.PROMPT,seed=42,max_new_tokens=512,
            recipe=run.RECIPE,question_order=list(bank),bank=bank,records=records)
        source.write_text(json.dumps(parent))
        with patch.object(run, 'TOTAL', 3): plan = run.prepare(root/'out', source)
        return source, parent, plan

    def test_exact_diffa_prompt_full_bank_only_two_deliveries_no_fitting(self):
        from diffa_gsm_vanilla import PROMPT
        self.assertEqual(run.PROMPT, PROMPT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, parent, plan = self.fixture(root)
            self.assertEqual(len(plan['recordings']), 6)
            self.assertEqual(plan['bank'], parent['bank'])
            self.assertEqual(plan['question_order'], parent['question_order'])
            self.assertEqual({r['condition'] for r in plan['recordings']}, {'neutral','angry'})
            self.assertEqual(plan['interventions'], [])
            self.assertFalse(any('collection' in k or 'fitting' in k for k in plan))
            Path(plan['recordings'][0]['audio_file']).write_bytes(b'changed')
            with patch.object(run, 'TOTAL', 3), self.assertRaisesRegex(ValueError, 'Source audio'):
                run.select(source)

    def test_full_source_must_have_both_deliveries_and_frozen_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, parent, plan = self.fixture(root)
            parent['records'] = [r for r in parent['records'] if not (r['question_id']=='q2' and r['condition']=='angry')]
            source.write_text(json.dumps(parent))
            with patch.object(run, 'TOTAL', 3), self.assertRaisesRegex(ValueError, 'every question'):
                run.select(source)
            plan['prompt'] = 'Supply reasoning.'
            with self.assertRaisesRegex(ValueError, 'settings differ'): run.validate_plan(plan)

    def test_inventory_selection_needs_neither_other_emotions_nor_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, parent, plan = self.fixture(root)
            class Registry:
                def neutral_ids(self, task): return list(parent['bank'])
                def choose(self, task, q, style): return self.options[task,q,style][0]
            registry = Registry(); registry.root = root; registry.options = {}
            for row in plan['recordings']:
                q, s = row['question_id'], row['style']
                registry.options['gsm8k',q,s] = [dict(record=row,
                    question=parent['bank'][q]['question'],answer=parent['bank'][q]['answer'])]
            with patch.object(run, 'SOURCES', ()), patch.object(run, 'TOTAL', 3), \
                    patch.object(run, 'Registry', return_value=registry):
                bank, order, records, receipt = run.select(audio_root=root)
            self.assertEqual(len(records), 6)
            self.assertEqual(set(bank), set(parent['bank']))

    def test_resume_no_capture_and_matched_official_scoring(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
            root = Path(tmp); source, parent, plan = self.fixture(root); out = root/'out'
            calls = []
            def generate(native, row, prompt, **kwargs):
                self.assertEqual(prompt, run.PROMPT)
                self.assertEqual(kwargs, dict(collect=False, interventions=[], max_new_tokens=512))
                if len(calls) == 3:
                    calls.append('interrupted'); raise KeyboardInterrupt
                calls.append((row['question_id'],row['condition']))
                return dict(response='#### '+row['reference_answer'], response_tokens=4, generation_seconds=.2)
            with patch('phi4_all_layer_diagnostic.generate', side_effect=generate):
                with self.assertRaises(KeyboardInterrupt): run.generate(out, plan, None)
                first = (out/'results.jsonl').read_bytes()
                run.report(out, plan)
                self.assertEqual(json.loads((out/'summary.json').read_text())['matched_questions'], 1)
                run.generate(out, plan, None); run.generate(out, plan, None)
            self.assertEqual(len(run.answers(out, plan)), 6)
            self.assertTrue((out/'results.jsonl').read_bytes().startswith(first))
            self.assertIn('Saved tokens=4', log.getvalue())
            run.report(out, plan)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['matched_questions'], 3)
            self.assertEqual({r['metric'] for r in summary['paired']},
                {'strict-match','flexible-extract','flexible-numeric','format-compliant'})
            self.assertTrue(all(r['baseline_correct']==3 and r['steered_correct']==3 for r in summary['paired']))

    def test_reuse_only_exact_phi_bare_answers_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, parent, plan = self.fixture(root); out = root/'out'
            rows = [dict(r, response='#### '+r['reference_answer'], response_tokens=4,
                run_config_sha256=run.sha(source),model='phi',prompt=run.PROMPT,seed=42,max_new_tokens=512,
                strength=0.,interventions=[]) for r in parent['records']]
            original = ''.join(json.dumps(r)+'\n' for r in rows)
            (source.parent/'deliveries.jsonl').write_text(original)
            self.assertEqual(run.reuse_night(out, plan, source), 6)
            self.assertEqual(run.reuse_night(out, plan, source), 0)
            self.assertEqual((source.parent/'deliveries.jsonl').read_text(), original)
            self.assertEqual(len(run.answers(out, plan)), 6)
            parent['prompt'] = 'Reason explicitly.'; source.write_text(json.dumps(parent))
            self.assertEqual(run.reuse_night(out, plan, source), 0)


if __name__ == '__main__': unittest.main()
