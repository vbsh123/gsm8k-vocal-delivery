from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_object_counting_angry_reasoning as run
from tests import test_object_counting_prompts as fixtures


class AngryReasoningTests(unittest.TestCase):
    def setUp(self):
        # The synthetic parent has its own valid fixture-source receipt.
        _,digest=fixtures.fixture_source()
        source=patch.object(run.base,'SOURCE_BLOB',digest)
        source.start()
        self.addCleanup(source.stop)

    def parent(self,root):
        source,selection=fixtures.ObjectCountingTests().selection(root)
        records={}
        for q in selection['question_order']:
            for s in run.base.STYLES:
                records[q,s]=dict(question_id=q,style=s,voice=run.base.VOICE,audio_file=str(source/(q+s+'.wav')),
                                  audio_sha256=q+s,answer=selection['bank'][q]['answer'])
        with patch.object(run.base,'source_records',return_value=records):
            plan=run.base.plan_for(selection,source)
        parent=root/'out'
        run.winners.freeze(parent/'run_config.json',plan)
        return parent,plan

    def test_reuses_angry_clips_and_exact_neutral_reasoning_prompt_without_parent_changes(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=self.parent(root)
            before={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
            new=run.prepare(parent,root/'extra')
            self.assertEqual(len(new['records']),250)
            self.assertEqual(new['question_order'],old['question_order'])
            self.assertEqual(new['prompt'],old['prompts']['neutral-reasoning'])
            self.assertEqual(new['prompts'],{run.CONDITION:run.base.REASONING_PROMPT})
            self.assertEqual(new['recipe'],old['recipe'])
            angry={r['question_id']:r for r in old['records'] if r['condition']=='angry'}
            for record in new['records']:
                original=angry[record['question_id']]
                self.assertEqual(record,dict(original,condition=run.CONDITION))
            self.assertEqual(before,{p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()})
            run.prepare(parent,root/'extra')

    def test_partial_parent_resume_and_matched_comparisons_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=self.parent(root)
            calls=[]
            class Backend:
                def generate(self,record,plan,capture=False):
                    calls.append((record['question_id'],record['condition'],plan['prompt'],capture))
                    return dict(response='The answer is '+record['reference_answer']+'.',response_tokens=8,audio_encoding={})
            backend=Backend()
            first=old['question_order'][0]
            original=Backend.generate
            def interrupt(self,record,plan,capture=False):
                if record['question_id']!=first:raise KeyboardInterrupt
                return original(self,record,plan,capture)
            with patch.object(Backend,'generate',interrupt):
                with self.assertRaises(KeyboardInterrupt):run.base.generate(old,parent,backend)
            parent_bytes={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
            output=root/'extra';plan=run.prepare(parent,output)
            run.base.generate(plan,output,backend)
            run.base.generate(plan,output,backend)
            newcalls=[r for r in calls if r[1]==run.CONDITION]
            self.assertEqual(len(newcalls),250)
            self.assertTrue(all(p==run.base.REASONING_PROMPT and not capture for q,c,p,capture in newcalls))
            run.report(output,parent)
            summary=json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(len(summary['comparisons']),6)
            self.assertEqual({v['reference'] for v in summary['comparisons']},{'neutral-reasoning','angry','neutral'})
            self.assertTrue(all(v['pairs']==1 and v['gain_pp']==0 for v in summary['comparisons']))
            self.assertEqual(summary['lengths'][run.CONDITION]['n'],250)
            self.assertEqual(summary['lengths']['neutral-reasoning']['n'],1)
            self.assertEqual(parent_bytes,{p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()})
            self.assertFalse(list(output.rglob('*.pt')))

    def test_changed_parent_prompt_or_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);parent,old=self.parent(root)
            plan=run.prepare(parent,root/'extra')
            old=json.loads((parent/'run_config.json').read_text())
            old['prompts']['neutral-reasoning']='A different prompt'
            (parent/'run_config.json').write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'source configuration'):run.prepare(parent,root/'other')
            with self.assertRaisesRegex(ValueError,'Parent configuration changed'):run.parent_plan(parent,plan)


if __name__=='__main__':unittest.main()
