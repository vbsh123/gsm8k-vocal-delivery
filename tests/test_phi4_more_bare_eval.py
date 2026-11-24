from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import phi4_more_bare_eval as run


class MorePhiBareTests(unittest.TestCase):
    def fixture(self, root, task):
        audio = root/task; audio.mkdir()
        bank, records = {}, []
        for i in range(4):
            q = f'{task}_{i:02d}'
            bank[q] = dict(question_id=q,question=f'Question {i}',answer='C' if task=='bbh5' else '2')
            if task=='bbh5': bank[q].update(original_input=f'Question {i}\nOptions:\n(A) X\n(B) Y\n(C) Z\n(D) V\n(E) W',original_target='(C)')
            for s in ('neutral','angry'):
                path = audio/(q+s+'.wav'); path.write_bytes((q+s).encode())
                records.append(dict(question_id=q,condition=s,style=s,voice='en-US-JennyNeural',
                    audio_file=str(path),audio_sha256=run.sha(path),reference_answer=bank[q]['answer']))
        order = sorted(bank); random.Random(42).shuffle(order)
        parent = dict(task=task,bank=bank,question_order=order,records=records)
        source = audio/'run_config.json'; source.write_text(json.dumps(parent))
        with patch.dict(run.TOTALS,{task:4}):
            plan = run.prepare(task,root/'out'/task,count=2,source=source)
        return source,parent,plan

    def test_prompts_match_diffa_and_fixed_subset_needs_only_two_styles(self):
        from diffa_vanilla_all import prompt_for
        for task in run.TOTALS:
            self.assertEqual(run.PROMPTS[task],prompt_for(task))
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source,parent,plan = self.fixture(root,task)
                self.assertEqual(plan['question_order'],parent['question_order'][:2])
                self.assertEqual(len(plan['recordings']),4)
                self.assertEqual({r['condition'] for r in plan['recordings']},{'neutral','angry'})
                self.assertFalse(any('collection' in k or 'fitting' in k for k in plan))
                self.assertEqual(plan['interventions'],[])
                plan['prompt'] = 'Supply reasoning.'
                with self.assertRaisesRegex(ValueError,'settings differ'): run.validate(plan)

    def test_exact_source_audio_gold_and_frozen_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source,parent,plan = self.fixture(root,'bbh5')
            with self.assertRaisesRegex(ValueError,'count differs'):
                run.prepare('bbh5',root/'out/bbh5',count=3,source=source)
            Path(plan['recordings'][0]['audio_file']).write_bytes(b'changed')
            with patch.dict(run.TOTALS,bbh5=4), self.assertRaisesRegex(ValueError,'recording/reference'):
                run.select('bbh5',2,source)

    def test_resume_no_capture_and_real_saved_answer_scoring_for_both_tasks(self):
        for task in run.TOTALS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
                root = Path(tmp); source,parent,plan = self.fixture(root,task); out = root/'out'/task
                calls = []
                def generate(native,row,prompt,**kwargs):
                    self.assertEqual(prompt,run.PROMPTS[task])
                    self.assertEqual(kwargs,dict(collect=False,interventions=[],max_new_tokens=512))
                    if len(calls)==3:
                        calls.append('interrupted'); raise KeyboardInterrupt
                    calls.append((row['question_id'],row['condition']))
                    return dict(response='The answer is (C).' if task=='bbh5' else '#### 2',
                        response_tokens=6,generation_seconds=.1)
                with patch('phi4_all_layer_diagnostic.generate',side_effect=generate):
                    with self.assertRaises(KeyboardInterrupt): run.generate(out,plan,None)
                    first = (out/'results.jsonl').read_bytes()
                    run.report(out,plan)
                    self.assertEqual(json.loads((out/'summary.json').read_text())['matched_questions'],1)
                    run.generate(out,plan,None); run.generate(out,plan,None)
                self.assertTrue((out/'results.jsonl').read_bytes().startswith(first))
                self.assertIn('Saved tokens=6',log.getvalue())
                self.assertEqual(len(run.answers(out,plan)),4)
                run.report(out,plan); summary = json.loads((out/'summary.json').read_text())
                self.assertEqual(summary['matched_questions'],2)
                self.assertTrue(all(r['baseline_correct']==2 and r['steered_correct']==2 for r in summary['paired']))
                metrics = {r['metric'] for r in summary['paired']}
                self.assertEqual(metrics,{'flexible-extract','strict-match'} if task=='bbh5' else
                    {'flexible-extract','strict-match','flexible-numeric','format-compliant'})

    def test_inventory_fallback_requires_no_happy_sad_or_emotion_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source,parent,plan = self.fixture(root,'multiarith')
            class Registry:
                def neutral_ids(self,task): return list(parent['bank'])
                def choose(self,task,q,s): return self.options[task,q,s][0]
            registry = Registry(); registry.root = root; registry.options = {}
            for row in parent['records']:
                q,s = row['question_id'],row['style']
                registry.options['multiarith',q,s] = [dict(record=row,
                    question=parent['bank'][q]['question'],answer=parent['bank'][q]['answer'])]
            with patch.dict(run.TOTALS,multiarith=4), patch.object(run,'Registry',return_value=registry):
                bank,order,records,receipt = run.select('multiarith',2,audio_root=root)
            self.assertEqual(order,parent['question_order'][:2]); self.assertEqual(len(records),4)


if __name__ == '__main__': unittest.main()
