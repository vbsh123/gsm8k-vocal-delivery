import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import phi4_bare_reverse as run


class BareReverseTests(unittest.TestCase):
    def fixture(self, task, total):
        bank, rows = {}, []
        for i in range(total):
            q = str(i)
            answer = 'C' if task == 'bbh5' else '2'
            bank[q] = dict(question='Some question',answer=answer,original_input='Question\nOptions:\n(C) example',
                           original_target='(C)')
            for s in ('neutral','angry'):
                rows.append(dict(question_id=q,condition=s,style=s,voice='Jenny',audio_file=q+s,
                                 audio_sha256=q+s,reference_answer=answer))
        return bank,list(bank),rows,{}

    def test_exact_disjoint_full_size_splits(self):
        for task,total in [('gsm8k',1319),('multiarith',580),('bbh5',250)]:
            module = run.gsm if task == 'gsm8k' else run.more
            with self.subTest(task=task), TemporaryDirectory() as tmp, patch.object(
                    module,'select',return_value=self.fixture(task,total)) as select:
                folder = Path(tmp)
                plan = run.prepare(task,folder)
                size = run.SIZES[task]
                self.assertEqual(len(set(plan['collection_ids']) | set(plan['evaluation_ids'])),size*2)
                self.assertEqual(len(plan['collection']),size)
                self.assertEqual(len(plan['evaluation']),size)
                self.assertEqual(plan['prompt'],run.PROMPTS[task])
                for q,p in zip(plan['collection_ids'],plan['collection']):
                    self.assertEqual([p[s]['question_id'] for s in ('neutral','angry')],[q,q])
                self.assertEqual([r['style'] for r in plan['evaluation']],['neutral']*size)
                self.assertEqual(run.prepare(task,folder)['collection_ids'],plan['collection_ids'])
                select.assert_called_once()

    def test_collect_sign_group_reverse_heldout_resume_and_score(self):
        task = 'multiarith'
        with TemporaryDirectory() as tmp, patch.dict(run.SIZES,{task:2}), patch.object(
                run.more,'select',return_value=self.fixture(task,4)), patch.object(run,'pilot_baselines',return_value={}):
            folder = Path(tmp)
            plan = run.prepare(task,folder)
            calls = []
            def fake(native,row,prompt,**kwargs):
                c,q = row['style'],row['question_id']
                self.assertEqual(prompt,run.PROMPTS[task])
                self.assertEqual(kwargs['max_new_tokens'],512)
                calls.append((q,c))
                trace = dict(response_tokens=3,response='#### 2')
                if kwargs.get('collect'):
                    self.assertEqual(kwargs['capture_layers'],[0,11,30])
                    self.assertIn(q,plan['collection_ids'])
                    # Unique delta for every captured layer/group.
                    diff = torch.arange(1,7).reshape(3,2,1).expand(3,2,3072).float()
                    trace.update(means=diff if c=='angry' else torch.zeros_like(diff),hook_forwards=2)
                else:
                    self.assertEqual(c,'neutral')
                    self.assertIn(q,plan['evaluation_ids'])
                    for intervention in kwargs['interventions']:
                        self.assertEqual(intervention['layer'],11)
                        self.assertEqual(intervention['group'],'answer')
                        self.assertEqual(intervention['strength'],-1)
                        self.assertTrue((intervention['direction'] == 3).all())
                return trace
            def interrupted(*args,**kwargs):
                if len(calls)==5:
                    raise KeyboardInterrupt
                return fake(*args,**kwargs)
            with patch.object(run.hooks,'generate',side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                run.evaluate(folder,plan,None)
            with patch.object(run.hooks,'generate',side_effect=fake):
                run.evaluate(folder,plan,None)
                run.evaluate(folder,plan,None)
            self.assertEqual(len(calls),8)
            vector=torch.load(folder/'style-vectors.pt',weights_only=True)['directions']
            for key,value in [('L0-audio',2),('L11-answer',3),('L30-answer',5)]:
                self.assertTrue((vector[key] == value).all())
            run.report(folder,plan)
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['matched'],2)
            self.assertEqual(summary['paired']['flexible-numeric']['baseline_correct'],2)
            self.assertEqual(summary['paired']['flexible-numeric']['gain_pp'],0)
