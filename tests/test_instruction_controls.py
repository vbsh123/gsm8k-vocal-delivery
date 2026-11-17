from contextlib import redirect_stdout
import io
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import collect_instruction_controls as controls
from tests.test_reasoning_geometry_audit import GeometryAuditTests


class InstructionControlTests(unittest.TestCase):
    def test_format_please_extension_reuses_all_old_controls_only100_new(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, shape = self.fixture(Path(d), 'phi4')
            with patch.object(controls, 'state_shape', return_value=shape):
                old = controls.prepare(args); (args.output/'states').mkdir()
                for i in range(100):
                    trace = torch.load(controls.state_path(args.parent, i, 'neutral-reasoning'), weights_only=True)
                    for c, prompt in old['prompts'].items():
                        torch.save(dict(trace, protocol=controls.PROTOCOL, condition=c, prompt=prompt),
                                   controls.state_path(args.output, i, c))
                out = Path(d)/'extension'; out.mkdir()
                extended = SimpleNamespace(model='phi4', parent=args.parent, output=out,
                                           extend_format_please=True, controls_source=args.output)
                plan = controls.prepare(extended)
                self.assertEqual(plan['new_forwards'], 100)
                self.assertEqual(len(plan['prompts']), 11)
                self.assertEqual(len(plan['reuse_controls']), 10)
                self.assertEqual(plan['parent']['pairs'], old['parent']['pairs'])
                self.assertIn(controls.FORMAT_PLEASE, plan['prompts']['format_please'])
                self.assertEqual(controls.control_path(plan, out, 0, 'polite'), controls.state_path(args.output, 0, 'polite'))
                self.assertEqual(controls.control_path(plan, out, 0, 'format_please'), controls.state_path(out, 0, 'format_please'))
                # The only new condition is stored in the extension folder.
                (out/'states').mkdir()
                for i in range(100):
                    t = torch.load(controls.state_path(args.parent, i, 'neutral-reasoning'), weights_only=True)
                    torch.save(dict(t, protocol=controls.PROTOCOL, condition='format_please', prompt=plan['prompts']['format_please']),
                               controls.state_path(out, i, 'format_please'))
                controls.summarize(plan, out)
                summary = json.loads((out/'summary.json').read_text())
                self.assertEqual(len(summary['complete_controls']), 11)
                self.assertTrue(any(r['reference'] == 'polite' and r['instruction'] == 'format_please' for r in summary['rows']))
                self.assertTrue(any(r['reference'] == 'format' and r['instruction'] == 'format_please' for r in summary['rows']))
                self.assertEqual(plan, controls.prepare(extended))
                reused = controls.state_path(args.output, 0, 'polite')
                with reused.open('ab') as stream: stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'Reused control/config changed'): controls.prepare(extended)

    def test_summary_runs_read_only_while_collector_holds_lock(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, shape = self.fixture(Path(d), 'phi4')
            with patch.object(controls, 'state_shape', return_value=shape):
                controls.prepare(args)
                lock = args.output/'queue.lock'
                with lock.open('w') as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    argv = ['collect_instruction_controls.py', '--model', 'phi4', '--mode', 'summary',
                            '--parent', str(args.parent), '--output', str(args.output)]
                    with patch('sys.argv', argv): controls.main()
                self.assertFalse((args.output/'summary.json').exists())
                self.assertFalse((args.output/'instruction-cosines.csv').exists())

    def test_reference_replay_detects_model_runtime_drift(self):
        expected = torch.tensor([[1., 2., 3.]])
        self.assertEqual(controls.compare_replay(expected.clone(), expected)['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'Native reference replay differs'):
            controls.compare_replay(expected+1, expected)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            controls.compare_replay(expected*float('nan'), expected)

    def fixture(self, root, model):
        parent = root/'parent'; parent.mkdir()
        shape = GeometryAuditTests().fixture(parent, model)
        path = parent/'run_config.json'; p = json.loads(path.read_text())
        p['reasoning_prompt'] = "Solve the audio. Let's think step by step. End with #### <number>."
        path.write_text(json.dumps(p))
        for i in range(100):
            for c in controls.CONDITIONS:
                path = controls.state_path(parent, i, c)
                t = torch.load(path, weights_only=True)
                t['prompt'] = 'direct' if c == 'neutral-answer-only' else p['reasoning_prompt']
                torch.save(t, path)
        output = root/'controls'; output.mkdir()
        return SimpleNamespace(model=model, parent=parent, output=output), shape

    def test_prompts_preserve_task_and_suffix_and_remove_reasoning_only(self):
        source = 'Solve audio. Give concise step-by-step reasoning with only the necessary calculations. End #### <number>.'
        prompts = controls.prompts(source)
        self.assertEqual(len(prompts), 10)
        for prompt in prompts.values():
            self.assertTrue(prompt.startswith('Solve audio. '))
            self.assertTrue(prompt.endswith(' End #### <number>.'))
        self.assertNotIn('reasoning', prompts['concise'])
        self.assertEqual(prompts['no_extra_instruction'], 'Solve audio.  End #### <number>.')
        with self.assertRaises(ValueError): controls.prompts('No known reasoning sentence')

    def test_exact_original100_frozen_parent_receipts_and_summary(self):
        for model in ('diffa', 'phi4'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
                args, shape = self.fixture(Path(d), model)
                with patch.object(controls, 'state_shape', return_value=shape):
                    plan = controls.prepare(args)
                    self.assertEqual(plan['parent']['pairs'], json.loads((args.parent/'run_config.json').read_text())['pairs'])
                    self.assertEqual(plan['new_forwards'], 1000)
                    self.assertEqual(plan, controls.prepare(args))
                    (args.output/'states').mkdir()
                    for i in range(100):
                        old = torch.load(controls.state_path(args.parent, i, 'neutral-reasoning'), weights_only=True)
                        for c in ('concise', 'no_extra_instruction'):
                            saved = dict(old, protocol=controls.PROTOCOL, condition=c, prompt=plan['prompts'][c])
                            if c == 'no_extra_instruction':
                                if model == 'diffa': saved['states'] = saved['states']-1
                                else:
                                    saved['answer'] = saved['answer']-1; saved['audio'] = saved['audio']-1
                            torch.save(saved, controls.state_path(args.output, i, c))
                    controls.summarize(plan, args.output)
                    summary = json.loads((args.output/'summary.json').read_text())
                    self.assertEqual(set(summary['complete_controls']), {'concise', 'no_extra_instruction'})
                    row = next(r for r in summary['rows'] if r['reference'] == 'answer_only'
                               and r['instruction'] == 'original_reasoning' and r['group'] == 'answer' and r['layer'] == 26)
                    self.assertAlmostEqual(row['cosine'], -.6)
                    self.assertAlmostEqual(row['signed_projection_coefficient'], -.48)
                    self.assertTrue(any(r['reference'] == 'bare_task' for r in summary['rows']))
                    self.assertFalse(any(r['reference'] == 'bare_task' and r['instruction'] == 'no_extra_instruction' for r in summary['rows']))
                    path = controls.state_path(args.parent, 0, 'neutral-reasoning')
                    t = torch.load(path, weights_only=True); t['prompt'] = 'changed'; torch.save(t, path)
                    with self.assertRaisesRegex(ValueError, 'Parent state changed'): controls.prepare(args)


if __name__ == '__main__': unittest.main()
