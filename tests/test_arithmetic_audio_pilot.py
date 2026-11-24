from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_arithmetic_audio_pilot as run


def source_fixture():
    examples = [dict(input=f'(({i%10} - -2) * (3 + -4)) =', target=str(-(i%10+2))) for i in range(250)]
    data = json.dumps(dict(examples=examples)).encode()
    digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    return data, digest


class ArithmeticPilotTests(unittest.TestCase):
    def prepared(self, root):
        source = root/'audio'; source.mkdir()
        data, digest = source_fixture()
        (source/'bbh-source.json').write_bytes(data)
        with patch.object(run, 'SOURCE_BLOB', digest):
            selection = run.prepare(source, root/'out')
        return source, selection

    def test_spoken_grouping_unary_minus_and_precedence(self):
        expression = '((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) ='
        spoken = run.spoken_question(expression)
        self.assertEqual(spoken.count('open parenthesis'), 3)
        self.assertEqual(spoken.count('close parenthesis'), 3)
        self.assertEqual(spoken.count('negative'), 5)
        self.assertEqual(spoken.count('minus'), 1)
        self.assertIn('nine times five', spoken)
        self.assertNotIn(',,', spoken)
        self.assertNotRegex(spoken, r'[0-9()*=]')
        self.assertEqual(run.expression_value(expression), 24)
        self.assertIn('minus negative two', run.spoken_question('(3 - -2) ='))
        self.assertEqual(run.expression_value('(2 + 3 * 4) ='), 14)
        for invalid in ('(3 / 2) =', 'abs(-3) =', '3 ** 2 =', '3 + 2', '3.5 + 2 ='):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run.spoken_question(invalid)

    def test_selection_fixed50_source_targets_and_no_filters(self):
        data, digest = source_fixture()
        with patch.object(run, 'SOURCE_BLOB', digest):
            questions = run.source_questions(data)
            self.assertEqual(questions, run.source_questions(data))
            self.assertEqual(len(questions), 50)
            self.assertEqual(len({q['question_id'] for q in questions}), 50)
            self.assertEqual(questions[0]['source_index'], 44)
            for q in questions:
                original = json.loads(data)['examples'][q['source_index']]
                self.assertEqual(q['original_input'], original['input'])
                self.assertEqual(q['answer'], original['target'])
            with self.assertRaisesRegex(ValueError, 'Git blob'):
                run.source_questions(data+b' ')
        modified = json.loads(data); modified['examples'][-1]['target'] = '123456'
        bad = json.dumps(modified).encode()
        bad_digest = hashlib.sha1(b'blob '+str(len(bad)).encode()+b'\0'+bad).hexdigest()
        with patch.object(run, 'SOURCE_BLOB', bad_digest), self.assertRaisesRegex(ValueError, 'target mismatch'):
            run.source_questions(bad)

    def test_remaining_is_exact_complement_and_keeps_pilot_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, pilot = self.prepared(root)
            replica_audio = root/'replica-audio'
            replica_audio.mkdir()
            data, digest = source_fixture()
            (replica_audio/'bbh-source.json').write_bytes(data)
            with patch.object(run, 'SOURCE_BLOB', digest):
                replica = run.prepare(replica_audio, root/'replica', remaining=True)
            first, rest = set(pilot['bank']), set(replica['question_order'])
            self.assertEqual((len(first), len(rest), len(first & rest)), (50, 200, 0))
            self.assertEqual(first | rest, {f'bbh_arithmetic_{i:03d}' for i in range(250)})
            self.assertEqual(replica['protocol'], run.REPLICATION)
            for key in ('source_git_blob', 'prompts', 'conditions', 'styles', 'voice', 'azure_intensity',
                        'recipe', 'max_new_tokens', 'primary_metric'):
                self.assertEqual(pilot[key], replica[key])

    def test_interaction_is_question_paired_and_uses_complete_quartets(self):
        def result(x):
            return {'metrics': {'flexible-extract': x}}
        # Per-question contrasts are +1, -1 and +2. A fourth incomplete row is ignored.
        groups = {c: {i: result(x) for i, x in enumerate(xs)}
                  for c, xs in zip(run.CONDITIONS, [[0,0,1,1], [0,1,0], [0,0,0], [1,0,1]])}
        stats = run.accuracy_interaction(groups, 'flexible-extract')
        self.assertEqual(stats['pairs'], 3)
        self.assertAlmostEqual(stats['interaction_pp'], 200/3)
        self.assertLess(stats['ci95_pp'][0], stats['interaction_pp'])
        self.assertGreater(stats['ci95_pp'][1], stats['interaction_pp'])
        self.assertTrue(0 < stats['p'] < 1)

    def test_four_conditions100_wavs200_answers_and_both_styles_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, selection = self.prepared(root)
            records = {(q, s):dict(question_id=q, style=s, voice=run.base.VOICE,
                audio_file=str(source/(q+s+'.wav')), audio_sha256=q+s)
                for q in selection['question_order'] for s in run.STYLES}
            with patch.object(run, 'source_records', return_value=records):
                plan = run.plan_for(selection, source)
            self.assertEqual(len(plan['records']), 200)
            self.assertEqual(len({r['audio_file'] for r in plan['records']}), 100)
            self.assertEqual(set(r['style'] for r in plan['records']), {'neutral','angry'})
            for i in range(50):
                neutral, angry, reasoning, angry_reasoning = plan['records'][i*4:i*4+4]
                self.assertEqual(neutral['audio_file'], reasoning['audio_file'])
                self.assertEqual(neutral['audio_sha256'], reasoning['audio_sha256'])
                self.assertEqual(angry['style'], 'angry')
                self.assertEqual(angry['audio_file'], angry_reasoning['audio_file'])
                self.assertEqual(angry['audio_sha256'], angry_reasoning['audio_sha256'])
                self.assertEqual(run.PROMPTS[angry_reasoning['condition']], run.REASONING_PROMPT)
            self.assertEqual(run.REASONING_PROMPT.replace(run.REASONING_SENTENCE+' ', ''), run.BARE_PROMPT)

    def test_tts_only_requested_styles_and_prepare_without_paid_calls(self):
        with patch.object(run.subprocess, 'run') as child:
            run.synthesize(Path('/tmp/arithmetic-audio-test'), prepare_only=True)
        command = child.call_args.args[0]
        self.assertEqual(command[command.index('--num-examples')+1], '50')
        self.assertEqual(command[command.index('--styles')+1:command.index('--output')], ['neutral','angry'])
        self.assertIn('--prepare-only', command)

    def test_interruption_resume_and_partial_matched_scoring(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); source, selection = self.prepared(root)
            q = selection['question_order'][0]
            records = [dict(question_id=q, condition=c, style=c.removesuffix('-reasoning'),
                voice=run.base.VOICE, audio_file='test.wav', audio_sha256=c,
                reference_answer=selection['bank'][q]['answer']) for c in run.CONDITIONS]
            # Represent another neutral answer without its test conditions yet.
            q2 = selection['question_order'][1]
            records.append(dict(records[0], question_id=q2, reference_answer=selection['bank'][q2]['answer']))
            plan = dict(selection, records=records)
            output = root/'out'; run.winners.freeze(output/'run_config.json', plan)
            calls = []
            class Backend:
                def generate(self, record, scoped, capture=False):
                    calls.append((record['question_id'], record['condition']))
                    self_case.assertEqual(scoped['prompt'], run.PROMPTS[record['condition']])
                    self_case.assertFalse(capture)
                    return dict(response='The answer is '+record['reference_answer']+'.',
                        response_tokens=7, audio_encoding={})
            self_case = self
            original = Backend.generate
            def interrupted(self, record, scoped, capture=False):
                if record['condition']=='angry-reasoning': raise KeyboardInterrupt
                return original(self, record, scoped, capture)
            with patch.object(Backend, 'generate', interrupted):
                with self.assertRaises(KeyboardInterrupt): run.generate(plan, output, Backend())
            run.report(output)
            summary = json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(len(summary['comparisons']), 4)
            run.generate(plan, output, Backend()); run.generate(plan, output, Backend())
            self.assertEqual(len(calls), 5)
            run.report(output)
            summary = json.loads((output/'delivery-summary.json').read_text())
            self.assertEqual(len(summary['comparisons']), 8)
            self.assertTrue(all(c['pairs']==1 and c['gain_pp']==0 for c in summary['comparisons']))
            self.assertEqual(summary['lengths']['neutral']['n'], 2)
            self.assertEqual(summary['scorer']['task'], run.SCORER)
            rows = [json.loads(line) for line in (output/'deliveries.jsonl').read_text().splitlines()]
            rows[0]['audio_sha256'] = 'changed'
            (output/'deliveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'Saved answer differs'):
                run.base.raw_rows(output, plan)

    def test_upstream_negative_and_positive_answers(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            scorer = run.scoring_preflight(Path(directory))
            values = scorer.score(['The answer is -50.', 'The answer is 50.', '-50'],
                [dict(input='(-5 * 2 * 5) =', target='-50')]*3)
            self.assertEqual([r['correct'] for r in values['flexible-extract']], [True,False,True])
            self.assertEqual([r['correct'] for r in values['strict-match']], [True,False,False])


if __name__ == '__main__':
    unittest.main()
