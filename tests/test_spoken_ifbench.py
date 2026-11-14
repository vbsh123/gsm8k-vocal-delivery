from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import run_spoken_ifbench as run


class IFBenchTests(unittest.TestCase):
    def test_prepare_requests_exactly_three_styles_without_answer_labels(self):
        example = dict(key='1', prompt='Use the word apple twice.',
            instruction_id_list=['count:keywords_multiple'], kwargs=[dict(keyword='apple', frequency=2)])
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); upstream = root/'upstream'
            for name in run.FILES:
                path = upstream/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source')
            args = SimpleNamespace(source=root/'audio', num_examples=0, mode='prepare')
            with patch.object(run, 'UPSTREAM', upstream), patch.object(run.subprocess, 'run') as tts:
                run.prepare(args, [example])
            row = json.loads((args.source/'questions.jsonl').read_text())
            self.assertEqual(row['question'], example['prompt'])
            self.assertEqual(row['kwargs'], example['kwargs'])
            command = tts.call_args.args[0]
            self.assertEqual(command[command.index('--styles')+1:command.index('--output')], ['neutral', 'angry', 'happy'])
            self.assertIn('--prepare-only', command)
            self.assertNotIn('answer', row)

    def test_three_style_eligibility_does_not_require_sad(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d); plan = []
            for qid in ('short', 'long'):
                for style in run.STYLES:
                    name = f'{qid}-{style}.wav'
                    with wave.open(str(source/name), 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                        audio.writeframes(b'\0\0' * (3001 if qid == 'long' and style == 'angry' else 100))
                    plan.append(dict(question_id=qid, voice='Jenny', style=style, provider='azure', audio_path=name))
            (source/'plan.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in plan))
            tasks = run.recordings(source, run.STYLES)
            self.assertEqual(len(tasks), 3)
            self.assertEqual({row['style'] for row in tasks}, set(run.STYLES))
            self.assertEqual({row['question_id'] for row in tasks}, {'short'})

    def test_official_checker_mutation_cannot_change_original_kwargs(self):
        def checker(inp, answers):
            inp.kwargs[0].pop('unused', None)
            return SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True])
        evaluator = SimpleNamespace(InputExample=lambda **kwargs: SimpleNamespace(**kwargs),
            test_instruction_following_strict=checker, test_instruction_following_loose=checker)
        q = dict(key='1', prompt='Hello', instruction_id_list=['x'], kwargs=[dict(unused=None)])
        run.score(q, 'Hello', evaluator)
        self.assertEqual(q['kwargs'], [dict(unused=None)])

    def test_summary_requests_ifbench_verifiers_and_loose_primary(self):
        args = SimpleNamespace()
        evaluator = object()
        with patch.object(run, 'report') as report:
            run.summarize(args, evaluator)
        self.assertEqual(report.call_args.kwargs['styles'], ['neutral', 'angry', 'happy'])
        self.assertEqual(report.call_args.kwargs['benchmark'], 'IFBench')
        self.assertEqual(report.call_args.kwargs['revision'], run.REVISION)
        self.assertEqual(report.call_args.kwargs['primary_metric'], 'loose_prompt')
        self.assertIs(report.call_args.kwargs['evaluator'], evaluator)

    def test_preflight_failure_stops_before_any_audio_generation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('sys.argv', ['test', '--output', d]), \
                 patch.object(run, 'setup_official', return_value=[dict(key='1')]), \
                 patch.object(run, 'official_scorer'), \
                 patch.object(run, 'score', side_effect=RuntimeError('verifier unavailable')), \
                 patch.object(run, 'prepare') as prepare:
                with self.assertRaisesRegex(RuntimeError, 'verifier unavailable'):
                    run.main()
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
