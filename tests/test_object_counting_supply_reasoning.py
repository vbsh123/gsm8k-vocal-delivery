from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_object_counting_supply_reasoning as run
import run_object_counting_angry_reasoning as earlier
from tests import test_object_counting_angry_reasoning as fixtures


class SupplyReasoningTests(unittest.TestCase):
    def setUp(self):
        _,digest=fixtures.fixtures.fixture_source()
        source=patch.object(run.base,'SOURCE_BLOB',digest);source.start();self.addCleanup(source.stop)

    def test_exact_approved_prompt_same_audio_and_all250_ids(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=fixtures.AngryReasoningTests().parent(root)
            before={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
            new=run.prepare(parent,root/'new')
            expected=('Answer the question in the audio. Supply reasoning before the final answer. '
                      'End your response with a final answer line in the format:\nThe answer is <number>.')
            self.assertEqual(new['prompt'],expected)
            self.assertEqual(len(new['records']),500)
            self.assertEqual(len({r['question_id'] for r in new['records']}),250)
            self.assertEqual(new['question_order'],old['question_order'])
            previous={(r['question_id'],r['condition']):r for r in old['records']}
            for r in new['records']:
                style=run.CONDITIONS[r['condition']]
                self.assertEqual(r,dict(previous[r['question_id'],style],condition=r['condition']))
                self.assertEqual(new['prompts'][r['condition']],expected)
            self.assertEqual(new['recipe'],old['recipe'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()})
            run.prepare(parent,root/'new')

    def test_resume_paired_reporting_and_preview_include_both_new_conditions(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=fixtures.AngryReasoningTests().parent(root)
            first=old['question_order'][0];calls=[]
            class Backend:
                def generate(self,record,plan,capture=False):
                    calls.append((record['condition'],plan['prompt'],capture))
                    return dict(response='The answer is '+record['reference_answer']+'.',response_tokens=6,audio_encoding={})
            original=Backend.generate
            def first_question(self,r,p,capture=False):
                if r['question_id']!=first:raise KeyboardInterrupt
                return original(self,r,p,capture)
            with patch.object(Backend,'generate',first_question):
                with self.assertRaises(KeyboardInterrupt):run.base.generate(old,parent,Backend())
            old_extra=root/'earlier';extra=earlier.prepare(parent,old_extra)
            with patch.object(Backend,'generate',first_question):
                with self.assertRaises(KeyboardInterrupt):run.base.generate(extra,old_extra,Backend())
            parent_bytes={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
            previous_bytes={p.name:p.read_bytes() for p in old_extra.iterdir() if p.is_file()}
            output=root/'new';plan=run.prepare(parent,output)
            with patch.object(Backend,'generate',first_question):
                with self.assertRaises(KeyboardInterrupt):run.base.generate(plan,output,Backend())
            run.base.generate(plan,output,Backend());run.base.generate(plan,output,Backend())
            newcalls=[v for v in calls if v[0] in run.CONDITIONS]
            self.assertEqual(len(newcalls),500)
            self.assertTrue(all(p==run.PROMPT and not capture for c,p,capture in newcalls))
            run.report(output,parent,old_extra)
            summary=json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(len(summary['comparisons']),10)
            emotion=[v for v in summary['comparisons'] if v['reference']=='neutral-supply-reasoning']
            self.assertTrue(all(v['pairs']==250 and v['gain_pp']==0 for v in emotion))
            self.assertTrue(all(v['pairs']==1 for v in summary['comparisons'] if v not in emotion))
            self.assertEqual(parent_bytes,{p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()})
            self.assertEqual(previous_bytes,{p.name:p.read_bytes() for p in old_extra.iterdir() if p.is_file()})
            result=subprocess.run(['python3','show_object_counting_outputs.py','--run',str(parent),
                '--extra',str(old_extra),'--supply-run',str(output),'--question-id',first],check=True,text=True,capture_output=True)
            self.assertIn('neutral-supply-reasoning | tokens=6',result.stdout)
            self.assertIn('angry-supply-reasoning | tokens=6',result.stdout)
            self.assertEqual(result.stdout.count('AUDIO:'),8)
            self.assertFalse(list(output.rglob('*.pt')))

    def test_changed_prompt_cannot_mix_with_existing_condition(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=fixtures.AngryReasoningTests().parent(root)
            output=root/'new';plan=run.prepare(parent,output)
            with patch.object(run,'PROMPT','Different reasoning wording'):
                with self.assertRaisesRegex(ValueError,'Frozen settings changed'):run.prepare(parent,output)
            old=json.loads((parent/'run_config.json').read_text());old['seed']=123
            (parent/'run_config.json').write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'Parent configuration changed'):run.parent_plan(parent,plan)


if __name__=='__main__':unittest.main()
