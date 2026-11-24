from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import run_bbh3_angry_x2 as run
from tests.test_emotion_block_followups import FollowupTests


class BBH3AngryX2Tests(unittest.TestCase):
    def test_bbh5_uses_its_own_bare_vector_prompt_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            primary = FollowupTests().make_cases(root, ('bbh5',))
            task = primary/'diffa/bbh5'
            for emotion in ('happy', 'sad'): shutil.rmtree(task/emotion)
            old = json.loads((task/'angry/L31-answer-a1/run_config.json').read_text())
            output = root/'x2'
            job = run.prepare(primary, output, 'bbh5')
            self.assertEqual(job['source']['name'], 'bbh5')
            self.assertEqual(job['source']['vector_sha256'], old['source']['vector_sha256'])
            self.assertEqual(job['target']['records'], old['target']['records'])
            self.assertEqual(job['target']['prompt'], run.follow.night.catalog.prompt_for('bbh5'))
            self.assertEqual(job['strengths'], [0.,1.,2.])
            done = run.follow.validate(job, output)
            self.assertEqual(len(done), 22)
            self.assertEqual({r['strength'] for r in done.values()}, {0.,1.})
            self.assertEqual(run.prepare(primary, output, 'bbh5'), job)
            x4 = run.prepare(primary, root/'x4', 'bbh5', 4.)
            self.assertEqual(x4['strengths'], [0.,1.,4.])
            self.assertEqual(x4['source'], job['source'])
            self.assertEqual(x4['target'], job['target'])
            self.assertEqual(len(run.follow.validate(x4, root/'x4')), 22)
            self.assertEqual(run.prepare(primary, root/'x4', 'bbh5', 4.), x4)
            x25 = run.prepare(primary, root/'x2.5', 'bbh5', 2.5)
            self.assertEqual(x25['strengths'], [0.,1.,2.5])
            self.assertEqual(x25['source'], job['source'])
            self.assertEqual(x25['target'], job['target'])
            existing = run.follow.validate(x25, root/'x2.5')
            self.assertEqual(len(existing), 22)
            self.assertEqual(run.prepare(primary, root/'x2.5', 'bbh5', 2.5), x25)
            with (root/'x2.5/results.jsonl').open('a') as stream:
                for row in existing.values():
                    if row['strength'] != 1.: continue
                    stream.write(json.dumps(dict(row, strength=2.5, condition='L31-answer-a2.5',
                        interventions=[dict(layer=31,group='answer',strength=2.5)]))+'\n')
            self.assertEqual(len(run.follow.validate(x25, root/'x2.5')), 33)
            self.assertEqual(json.loads((output/'run_config.json').read_text())['strengths'], [0.,1.,2.])
            with self.assertRaisesRegex(ValueError, 'Expected BBH3 or BBH5'):
                run.prepare(primary, output, 'svamp')

    def test_only_angry_existing_vector_and_same_holdout_with_zero_and_one_reuse(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            primary = FollowupTests().make_cases(root, ('bbh3',))
            task = primary/'diffa/bbh3'
            # Other emotion checkpoints and BBH5 are not prerequisites.
            for emotion in ('happy', 'sad'): shutil.rmtree(task/emotion)
            old = json.loads((task/'angry/L31-answer-a1/run_config.json').read_text())
            old_results = (task/'angry/L31-answer-a1/results.jsonl').read_bytes()
            output = root/'x2'
            job = run.prepare(primary, output)
            self.assertEqual(job['strengths'], [0.,1.,2.])
            self.assertEqual(job['layer'], 31)
            self.assertEqual(job['group'], 'answer')
            self.assertEqual(job['source']['vector_sha256'], old['source']['vector_sha256'])
            self.assertEqual(job['target']['records'], old['target']['records'])
            self.assertEqual(job['target']['prompt'], old['target']['prompt'])
            self.assertFalse(set(job['source']['extraction_ids']) & {r['question_id'] for r in job['target']['records']})
            done = run.follow.validate(job, output)
            self.assertEqual(len(done), 22)
            self.assertEqual({r['strength'] for r in done.values()}, {0.,1.})
            before = (output/'results.jsonl').read_bytes()
            self.assertEqual(run.prepare(primary, output), job)
            self.assertEqual((output/'results.jsonl').read_bytes(), before)
            self.assertEqual((task/'angry/L31-answer-a1/results.jsonl').read_bytes(), old_results)
            # Simulate completed x2 responses and check the real cache validator.
            with (output/'results.jsonl').open('a') as stream:
                for row in done.values():
                    if row['strength'] != 1.: continue
                    changed = dict(row, strength=2., condition='L31-answer-a2',
                        interventions=[dict(layer=31,group='answer',strength=2.)])
                    stream.write(json.dumps(changed)+'\n')
            self.assertEqual(len(run.follow.validate(job, output)), 33)


if __name__ == '__main__':
    unittest.main()
