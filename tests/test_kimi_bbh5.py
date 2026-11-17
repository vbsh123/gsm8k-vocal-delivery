import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import torch
import run_kimi_bbh5 as run


class KimiBBH5Tests(unittest.TestCase):
    def fixture(self, root):
        source = root/'audio'; source.mkdir(); output = root/'out'; output.mkdir()
        rows = [dict(input=f'Order these five books. Question {i}.\nOptions:\n'+
                     '\n'.join(f'({c}) Book {c}' for c in 'ABCDE'), target='(E)') for i in range(250)]
        data = json.dumps(dict(examples=rows)).encode()
        (source/'logical_deduction_five_objects-source.json').write_bytes(data)
        args = SimpleNamespace(source=source, output=output, audio_sources=[])
        return args, data

    def test_pinned_source_disjoint_half_and_five_choice_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d))
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                selection = run.select(args)
                self.assertEqual(selection, run.select(args))
            self.assertEqual((len(selection['collection']), len(selection['test'])), (125, 125))
            self.assertFalse({q['question_id'] for q in selection['collection']} & {q['question_id'] for q in selection['test']})
            self.assertIn('five choices A, B, C, D, E', selection['prompt'])
            self.assertIn('The answer is (<letter>)', selection['prompt'])
            self.assertEqual(selection['layer'], 17)
            self.assertTrue(all(q['original_target'] == '(E)' and q['answer'] == 'E'
                                for q in selection['collection']+selection['test']))

    def plan(self, args, data):
        with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
            selection = run.select(args)
        with wave.open(str(args.source/'clip.wav'), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b'\0\0'*160)
        rows = [dict(q, provider='azure', voice='Jenny', style=s, audio_path='clip.wav')
                for q in selection['collection']+selection['test'] for s in ('neutral', 'angry')]
        (args.source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tree = ast.parse(Path(run.__file__).with_name('evaluate_kimi_audio.py').read_text())
        names = {'MODEL', 'REVISION', 'SOURCE_REVISION', 'TOKENIZER_MODEL', 'TOKENIZER_REVISION'}
        pins = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}
        with patch.dict(sys.modules, {'evaluate_kimi_audio': SimpleNamespace(**pins)}):
            return run.prepare(args, selection)

    def test_both_prompts_keep_same_split_and_separate_protocols(self):
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d))
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                old = run.select(args)
                args.explicit_reasoning = True
                with self.assertRaisesRegex(ValueError, 'Settings changed'):
                    run.select(args)
                args.output = Path(d)/'new'; args.output.mkdir()
                new = run.select(args)
            self.assertEqual(old['collection'], new['collection'])
            self.assertEqual(old['test'], new['test'])
            self.assertNotEqual(old['protocol'], new['protocol'])
            self.assertEqual(new['prompt'], old['prompt'].replace("Let's think step by step.",
                'Give concise step-by-step reasoning with only the necessary deductions.'))
            self.assertIn('The answer is (<letter>)', new['prompt'])

    def test_new_prompt_reaches_collection_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d)); args.explicit_reasoning = True
            plan = self.plan(args, data)
            self.assertEqual(plan['prompt'], run.EXPLICIT_PROMPT)
            self.assertEqual(plan['protocol'], run.PROTOCOL+'-explicit-reasoning')
            self.assertEqual(plan['selection']['prompt_variant'], 'explicit-reasoning')
            self.assertEqual(plan['steered_condition'], run.STEERED)

    def test_collection_references_reused_once_and_no_collection_steering(self):
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d)); plan = self.plan(args, data)
            self.assertEqual(plan['prompt'], run.PROMPT)
            self.assertNotIn('####', plan['prompt'])
            self.assertEqual(plan['protocol'], run.PROTOCOL)
            (args.output/'pairs').mkdir()
            for i, pair in enumerate(plan['collection']):
                torch.save(dict(protocol=run.PROTOCOL, question_id=pair['question_id'],
                    audio_sha256={s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')},
                    responses={'neutral': 'The answer is (E).', 'angry': 'The answer is (D).'},
                    response_tokens={'neutral': 6, 'angry': 6}), args.output/f'pairs/{i:05d}.pt')
            run.save_collection_references(plan, args.output)
            run.save_collection_references(plan, args.output)
            rows = run.shared.read_rows(args.output/'results.jsonl')
            self.assertEqual(len(rows), 250)
            self.assertEqual({r['condition'] for r in rows}, {'neutral', 'angry'})
            self.assertTrue(all(r['reused_from_collection'] for r in rows))

    def test_report_full_delivery_and_heldout_steering_with_all_five_format_choices(self):
        with tempfile.TemporaryDirectory() as d:
            args, data = self.fixture(Path(d)); plan = self.plan(args, data)
            train, test = plan['collection'][0], plan['test'][0]
            rows = []
            for pair, conditions in ((train, ('neutral', 'angry')), (test, ('neutral', 'angry', run.STEERED))):
                for c in conditions:
                    row = pair['angry' if c == 'angry' else 'neutral']
                    rows.append(dict(row, condition=c, prompt=run.PROMPT, response_tokens=6,
                                     response='The answer is (E).' if c != 'neutral' else 'The answer is (D).'))
            path = args.output/'results.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); before = path.read_bytes()
            run.report(args.output)
            summary = json.loads((args.output/'summary.json').read_text())
            all_rows = [r for r in summary['accuracy'] if r['cohort'] == 'all-250-delivery']
            held = [r for r in summary['accuracy'] if r['cohort'] == 'held-out-125-steering']
            self.assertEqual({r['n'] for r in all_rows}, {2})
            self.assertEqual({r['n'] for r in held}, {1})
            self.assertTrue(all(r['correct']['strict-format-extractable'] == r['n'] for r in all_rows+held))
            self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
