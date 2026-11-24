from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_aqua_audio_pilot as run


def fixture():
    rows = [dict(question=f'How many objects in set {i}?', options=[f'{l}){k}' for k,l in enumerate('ABCDE')],
                 correct='B', rationale='SECRET REFERENCE SOLUTION') for i in range(254)]
    data = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    return data, blob


class AquaPilotTests(unittest.TestCase):
    def prepare(self, root):
        source = root/'audio'; source.mkdir()
        data, blob = fixture(); (source/'aqua-test-source.jsonl').write_bytes(data)
        with patch.object(run, 'SOURCE_BLOB', blob): selection = run.prepare(source, root/'out')
        return source, selection

    def test_fixed_unfiltered_source_and_no_rationale_in_inputs(self):
        data, blob = fixture()
        with patch.object(run, 'SOURCE_BLOB', blob):
            selected = run.source_questions(data)
            self.assertEqual(selected, run.source_questions(data))
            self.assertEqual(len(selected), 50)
            self.assertEqual(len({q['source_index'] for q in selected}), 50)
            self.assertEqual(selected[0]['source_index'], 44)
            self.assertNotIn('SECRET', json.dumps(selected))
            self.assertEqual(selected[0]['answer'], 'B')
            self.assertEqual(selected[0]['question'].count('Option '), 5)
            with self.assertRaisesRegex(ValueError, 'Git blob'): run.source_questions(data+b' ')

    def test_explicit_choice_parser_and_strict_format_are_separate(self):
        for text in ['The answer is (C).', 'Work\nThe answer is **C**.', 'The answer is **(C)**.',
                     'The answer is (B).\nCorrecting myself:\nThe answer is (C).']:
            self.assertTrue(run.score(text, 'C')['metrics']['choice-accuracy'])
        for text in ['Option A is wrong. Option C is right.', '(C)', 'The answer is C or D.',
                     'The answer is not determinable.', 'The answer is (F).']:
            self.assertIsNone(run.choice(text))
        scored = run.score('The answer is **C**.', 'C')
        self.assertFalse(scored['metrics']['format-compliant'])
        self.assertFalse(scored['metrics']['strict-final-line'])
        self.assertTrue(run.score('Work\nThe answer is (C).', 'C')['metrics']['strict-final-line'])
        wrong = run.score('The answer is (B).', 'C')['metrics']
        self.assertTrue(wrong['format-compliant']); self.assertFalse(wrong['choice-accuracy'])

    def test_spoken_notation_no_silent_math_repairs(self):
        rendered = run.speak('If x<0, y>0 and |x^3| > |y^2|, find -x + 20% of $250.')
        for phrase in ('less than', 'greater than', 'vertical bar', 'to the power of', 'minus x',
                       'plus', 'percent', '250 dollars'): self.assertIn(phrase, rendered)
        self.assertNotRegex(rendered, r'[|^<>=*%$]')
        self.assertEqual(run.speak('5C3*4C3'), '5 choose 3 times 4 choose 3')
        self.assertEqual(run.speak('a2'), 'a 2')
        self.assertIn('yo-yo', run.speak('3-letter codes and yo-yo'))
        self.assertIn('3-letter', run.speak('3-letter codes and yo-yo'))
        self.assertIn('slash', run.speak('w/25'))

    def test100_clips_records_same_prompt_and_prepare_is_unpaid(self):
        with tempfile.TemporaryDirectory() as directory:
            source, selection = self.prepare(Path(directory))
            records = {(q,c):dict(question_id=q, style=c, voice=run.base.VOICE,
                audio_file=q+c+'.wav', audio_sha256=q+c) for q in selection['question_order'] for c in run.CONDITIONS}
            with patch.object(run, 'source_records', return_value=records): plan = run.plan_for(selection, source)
            self.assertEqual(len(plan['records']), 100)
            self.assertEqual(len({r['audio_file'] for r in plan['records']}), 100)
            self.assertEqual(set(plan['prompts'].values()), {run.PROMPT})
            with patch.object(run.subprocess, 'run') as call: run.synthesize(source, prepare_only=True)
            cmd = call.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--styles')+1:cmd.index('--output')], ['neutral','angry'])
            self.assertIn('--prepare-only', cmd)

    def test_resume_preserves_saved_answers_and_pairs_only_completed_ids(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as log:
            root = Path(directory); source, selection = self.prepare(root)
            qids = selection['question_order'][:2]
            records = [dict(question_id=q, condition=c, style=c, voice=run.base.VOICE, audio_file=q+c+'.wav',
                audio_sha256=q+c, reference_answer='B') for q in qids for c in run.CONDITIONS]
            plan = dict(selection, records=records)
            output = root/'out'; run.winners.freeze(output/'run_config.json', plan)
            calls = []
            class Backend:
                def generate(self, record, scoped, capture=False):
                    if (record['question_id'], record['condition']) == (qids[1], 'angry') and len(calls) == 3:
                        calls.append('interrupted'); raise KeyboardInterrupt
                    calls.append((record['question_id'], record['condition']))
                    assert scoped['prompt'] == run.PROMPT and capture is False
                    return dict(response='The answer is (B).', response_tokens=6, audio_encoding={})
            with self.assertRaises(KeyboardInterrupt): run.generate(plan, output, Backend())
            first = (output/'deliveries.jsonl').read_text()
            run.report(output)
            summary = json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(summary['comparisons'][0]['pairs'], 1)
            run.generate(plan, output, Backend()); run.generate(plan, output, Backend())
            self.assertTrue((output/'deliveries.jsonl').read_text().startswith(first))
            self.assertEqual(len(run.base.raw_rows(output, plan)), 4)
            self.assertIn('Saved tokens=6', log.getvalue())
            run.report(output)
            summary = json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(summary['comparisons'][0]['pairs'], 2)
            preview = subprocess.run([sys.executable, 'show_aqua_outputs.py', '--run', str(output), '--count', '3', '--longest'],
                capture_output=True, text=True, check=True)
            self.assertIn('Longest angry', preview.stdout)
            self.assertEqual(preview.stdout.count('GOLD: B'), 2)
            tampered = json.loads(first.splitlines()[0]); tampered['prompt'] = 'changed'
            with (output/'deliveries.jsonl').open('a') as stream: stream.write(json.dumps(tampered)+'\n')
            with self.assertRaisesRegex(ValueError, 'differs'): run.base.raw_rows(output, plan)


if __name__ == '__main__': unittest.main()
