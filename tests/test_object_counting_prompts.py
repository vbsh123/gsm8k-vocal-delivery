from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_object_counting_prompts as run


def fixture_source():
    data=json.dumps(dict(examples=[dict(input=f'Objects question {i}?', target=str(2+i%17)) for i in range(250)])).encode()
    digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    return data,digest


class ObjectCountingTests(unittest.TestCase):
    def selection(self,root):
        source=root/'audio';source.mkdir()
        data,digest=fixture_source()
        (source/'bbh-source.json').write_bytes(data)
        with patch.object(run,'SOURCE_BLOB',digest):
            selection=run.prepare(source,root/'out')
        return source,selection

    def test_source_integrity_all250_unfiltered_and_only_reasoning_sentence_changes(self):
        data,digest=fixture_source()
        with patch.object(run,'SOURCE_BLOB',digest):
            questions=run.source_questions(data)
            self.assertEqual(questions,run.source_questions(data))
            with self.assertRaisesRegex(ValueError,'Git blob'):
                run.source_questions(data+b' ')
        self.assertEqual(len(questions),250)
        self.assertEqual(len({q['question_id'] for q in questions}),250)
        for q in questions:
            original=json.loads(data)['examples'][q['source_index']]
            self.assertEqual(q['question'],original['input'])
            self.assertEqual(q['answer'],original['target'])
        self.assertEqual(run.REASONING_PROMPT.replace(run.REASONING_SENTENCE+' ',''),run.BARE_PROMPT)
        self.assertNotIn('step',run.BARE_PROMPT)

    def test_five_conditions_reuse_identical_neutral_audio_and_all_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,selection=self.selection(root)
            records={}
            for q in selection['question_order']:
                for style in run.STYLES:
                    records[q,style]=dict(question_id=q,style=style,voice=run.VOICE,
                        audio_file=str(source/(q+style+'.wav')),audio_sha256=q+style,
                        answer=selection['bank'][q]['answer'])
            with patch.object(run,'source_records',return_value=records):
                plan=run.plan_for(selection,source)
            self.assertEqual(len(plan['records']),1250)
            self.assertEqual(len({r['audio_file'] for r in plan['records']}),1000)
            for i in range(250):
                chunk=plan['records'][i*5:(i+1)*5]
                self.assertEqual([r['condition'] for r in chunk],list(run.CONDITIONS))
                self.assertEqual(chunk[0]['audio_sha256'],chunk[1]['audio_sha256'])
                self.assertEqual(chunk[0]['audio_file'],chunk[1]['audio_file'])
            self.assertEqual(plan['recipe'],run.evaluation.night.vanilla.RECIPE)
            self.assertFalse((root/'out'/'angry').exists())

    def test_tts_prepare_uses_only_four_audio_styles_and_no_paid_calls(self):
        with patch.object(run.subprocess,'run') as child:
            run.synthesize(Path('/tmp/object-counting-test'),prepare_only=True)
        command=child.call_args.args[0]
        self.assertIn('--prepare-only',command)
        self.assertEqual(command[command.index('--num-examples')+1],'250')
        self.assertEqual(command[command.index('--styles')+1:command.index('--output')],list(run.STYLES))
        self.assertNotIn('neutral-reasoning',command)

    def test_generation_resume_and_report_keep_condition_specific_prompts(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);source,selection=self.selection(root)
            q=selection['question_order'][0]
            records=[]
            for condition in run.CONDITIONS:
                style='neutral' if condition=='neutral-reasoning' else condition
                records.append(dict(question_id=q,condition=condition,style=style,voice=run.VOICE,
                    audio_file='fake.wav',audio_sha256=style,reference_answer=selection['bank'][q]['answer']))
            plan=dict(selection,records=records)
            output=root/'out';run.winners.freeze(output/'run_config.json',plan)
            calls=[]
            class Backend:
                def generate(self,record,scoped,capture=False):
                    calls.append((record['condition'],scoped['prompt'],capture))
                    return dict(response='The answer is '+record['reference_answer']+'.',response_tokens=7,audio_encoding={})
            # Simulate an interrupted row sequence then resume with every condition.
            original=Backend.generate
            def interrupted(self,record,scoped,capture=False):
                if record['condition']=='happy':raise KeyboardInterrupt
                return original(self,record,scoped,capture)
            with patch.object(Backend,'generate',interrupted):
                with self.assertRaises(KeyboardInterrupt):run.generate(plan,output,Backend())
            run.generate(plan,output,Backend());run.generate(plan,output,Backend())
            self.assertEqual(len(calls),5)
            self.assertTrue(all(prompt==run.PROMPTS[c] and not capture for c,prompt,capture in calls))
            self.assertEqual(len(run.raw_rows(output,plan)),5)
            self.assertFalse(list(output.rglob('*.pt')))
            run.report(output)
            summary=json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(len(summary['comparisons']),8)
            self.assertEqual(set(summary['lengths']),set(run.CONDITIONS))
            self.assertTrue(all(r['pairs']==1 and r['gain_pp']==0 for r in summary['comparisons']))
            self.assertEqual(summary['protocol'],run.PROTOCOL)
            rows=[json.loads(l) for l in (output/'deliveries.jsonl').read_text().splitlines()]
            rows[0]['prompt']=run.BARE_PROMPT
            (output/'deliveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Saved answer differs'):run.raw_rows(output,plan)

    def test_real_upstream_numeric_and_format_filters(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            scorer=run.scoring_preflight(Path(directory))
            values=scorer.score(['The answer is 8.','8','The answer is eight.','The answer is 9.'],
                                [dict(input='Count?',target='8')]*4)
            self.assertEqual([v['correct'] for v in values['strict-match']],[True,False,False,False])
            # Preserve the pinned upstream behavior: its number-word fallback
            # does not recover this case. Do not silently rewrite responses.
            self.assertEqual([v['correct'] for v in values['flexible-extract']],[True,True,False,False])
            self.assertEqual(values['flexible-extract'][2]['prediction'],'[invalid]')


if __name__=='__main__':
    unittest.main()
