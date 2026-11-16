import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_missing_experiments as run


class RecoveryQueueTests(unittest.TestCase):
    def test_backup_copies_audio_answers_vectors_and_never_removes_saved_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); outputs = root/'outputs'; source = outputs/'azure-multiarith580'
            source.mkdir(parents=True); backup = root/'backup'
            for name in ('recording.wav', 'results.jsonl', 'style-vectors.pt'):
                (source/name).write_bytes(b'example')
            run.snapshot(outputs, backup)
            for name in ('recording.wav', 'results.jsonl', 'style-vectors.pt'):
                self.assertEqual((backup/source.name/name).read_bytes(), b'example')
                self.assertTrue((source/name).exists())
            (source/'results.jsonl').write_bytes(b'updated answers')
            run.snapshot(outputs, backup)
            self.assertEqual((backup/source.name/'results.jsonl').read_bytes(), b'updated answers')
            (source/'recording.wav').unlink()
            run.snapshot(outputs, backup)
            self.assertTrue((backup/source.name/'recording.wav').exists())
            with self.assertRaises(ValueError):
                run.snapshot(outputs, outputs/'backups')

    def test_queue_uses_original_scripts_and_defaults_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            with patch.object(sys, 'argv', ['queue', '--backup-root', d]), \
                 patch.object(run.os, 'chdir'), patch.object(run, 'snapshot'), \
                 patch.object(run, 'run_stage', side_effect=lambda name, command, backup: calls.append(command)):
                run.main()
            stages = calls[5:]
            self.assertEqual(stages, [
                [sys.executable, 'run_multiarith_mean.py'],
                ['bash', 'run_spoken_ifeval.sh', '--mode', 'generate'],
                [sys.executable, 'run_spoken_ifeval.py', '--mode', 'evaluate'],
                [sys.executable, 'steer_ifeval_mean.py']])
            self.assertIn('bbh-audio-models', calls[0])

    def test_failure_stops_queue_before_next_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def stage(name, command, backup):
                calls.append(command)
                if 'run_multiarith_mean.py' in command:
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(sys, 'argv', ['queue', '--backup-root', d]), \
                 patch.object(run.os, 'chdir'), patch.object(run, 'snapshot'), \
                 patch.object(run, 'run_stage', side_effect=stage):
                with self.assertRaises(subprocess.CalledProcessError):
                    run.main()
            self.assertFalse(any('run_spoken_ifeval.sh' in c for c in calls))

    def test_stage_backs_up_child_outputs_before_returning(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            try:
                os.chdir(root)
                command = [sys.executable, '-c',
                    "from pathlib import Path; p=Path('outputs/multiarith580-angry-mean'); p.mkdir(parents=True); (p/'results.jsonl').write_text('{}\\n')"]
                run.run_stage('test', command, root/'backup')
                self.assertEqual((root/'backup/multiarith580-angry-mean/results.jsonl').read_text(), '{}\n')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
