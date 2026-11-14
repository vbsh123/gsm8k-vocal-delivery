from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import run_spoken_ifeval as run


class SpokenIFEvalTests(unittest.TestCase):
    def test_excludes_entire_question_if_one_delivery_is_long(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = []
            for qid in ('short', 'long'):
                for style in run.STYLES:
                    name = f'{qid}-{style}.wav'
                    with wave.open(str(root/name), 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(100)
                        audio.writeframes(b'\0\0' * (3001 if qid == 'long' and style == 'sad' else 100))
                    tasks.append(dict(provider='azure', question_id=qid, voice='Jenny',
                                      style=style, audio_path=name))
            (root/'plan.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
            eligible = run.recordings(root)
            self.assertEqual(len(eligible), 4)
            self.assertEqual({t['question_id'] for t in eligible}, {'short'})
            self.assertEqual(json.loads((root/'eligibility.json').read_text())['excluded_questions'], ['long'])

    def test_score_delegates_original_metadata_and_unmodified_response(self):
        calls = []
        def check(inp, responses):
            calls.append((inp, responses))
            return SimpleNamespace(follow_all_instructions=False, follow_instruction_list=[True, False])
        evaluator = SimpleNamespace(InputExample=lambda **kw: SimpleNamespace(**kw),
            test_instruction_following_strict=check, test_instruction_following_loose=check)
        example = dict(key=12, prompt='Use JSON.', instruction_id_list=['a', 'b'], kwargs=[{}, {}],
                       question_id='ifeval_12')
        result = run.score(example, 'Here is my answer: {}', evaluator)
        self.assertEqual(calls[0][1], {'Use JSON.': 'Here is my answer: {}'})
        self.assertEqual(calls[0][0].instruction_id_list, ['a', 'b'])
        self.assertEqual(result['strict'], dict(prompt=False, instructions=[True, False]))
        self.assertEqual(len(calls), 2)

    def test_summary_does_not_take_generation_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('sys.argv', ['test', '--mode', 'summary', '--output', d]), \
                 patch.object(run, 'report') as report, patch.object(run.fcntl, 'flock') as lock:
                run.main()
            report.assert_called_once()
            lock.assert_not_called()

    def test_prepare_preserves_prompt_and_constraints_without_api_calls(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); package = root/'official'; (package/'data').mkdir(parents=True)
            example = dict(key=42, prompt='Include exactly two bullets.',
                instruction_id_list=['detectable_format:number_bullet_lists'], kwargs=[dict(num_bullets=2)])
            (package/'data/input_data.jsonl').write_text(json.dumps(example)+'\n')
            args = SimpleNamespace(source=root/'audio', num_examples=0, mode='prepare')
            with patch.object(run, 'setup_official', return_value=package), \
                 patch.object(run.subprocess, 'run') as command:
                run.prepare(args)
            row = json.loads((args.source/'questions.jsonl').read_text())
            self.assertEqual(row['question'], example['prompt'])
            self.assertEqual(row['kwargs'], example['kwargs'])
            self.assertIn('--prepare-only', command.call_args.args[0])
            self.assertEqual(command.call_args.args[0][command.call_args.args[0].index('--styles')+1:
                command.call_args.args[0].index('--output')], run.STYLES)


if __name__ == '__main__':
    unittest.main()
