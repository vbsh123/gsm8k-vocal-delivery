from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from steer_audio import main, select_conditions


class FinalistResumeTests(unittest.TestCase):
    def test_selection_keeps_baseline_and_rejects_typo(self):
        original=[dict(name=n) for n in ('baseline','other','L31-answer-a1','L31-answer-a1-periodic2')]
        selected=select_conditions(original,['L31-answer-a1','L31-answer-a1-periodic2'])
        self.assertEqual([c['name'] for c in selected],['baseline','L31-answer-a1','L31-answer-a1-periodic2'])
        self.assertEqual(len(original),4)
        with self.assertRaises(ValueError):
            select_conditions(original,['typo'])

    def test_cached_finalists_skip_model_and_preserve_other_answers_and_config(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            vector=root/'vectors.pt';vector.write_bytes(b'not a tensor; must not be loaded')
            names=['baseline','other','L31-answer-a1','L31-answer-a1-periodic2']
            config=dict(recordings=[dict(clip_id='q')],settings=dict(
                seed=42,seed_policy='constant_before_every_generation',vectors=str(vector),
                vector_bytes=vector.stat().st_size,vector_mtime_ns=vector.stat().st_mtime_ns,
                conditions=[dict(name=n,interventions=[]) for n in names]))
            path=root/'run_config.json';path.write_text(json.dumps(config))
            result=root/'results.jsonl'
            result.write_text(''.join(json.dumps(dict(clip_id='q',condition=n,seed=42,correct=n!='baseline'))+'\n' for n in names))
            previous_config=path.read_bytes();previous_results=result.read_bytes()
            summary=main(['--config',str(path),'--output',str(root),'--conditions',*names[2:]])
            self.assertEqual([r['condition'] for r in summary],names[2:])
            self.assertEqual(path.read_bytes(),previous_config)
            self.assertEqual(result.read_bytes(),previous_results)


if __name__=='__main__':
    unittest.main()
