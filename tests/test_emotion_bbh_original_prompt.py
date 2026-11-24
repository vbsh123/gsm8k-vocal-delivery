from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import emotion_bbh_original_prompt as original
import tests.test_emotion_block_night as fixtures


class OriginalPromptTests(unittest.TestCase):
    def test_exact_historical_prompt_strings(self):
        from run_bbh_logical200 import PROMPT
        from run_reasoning_transfer import BBH_PROMPT
        self.assertEqual(original.original_prompt('bbh3'),PROMPT)
        self.assertEqual(original.original_prompt('bbh5'),BBH_PROMPT)
        with self.assertRaises(ValueError):
            original.original_prompt('gsm8k')

    def test_reuses_all_emotion_fitting_ids_but_recollects_and_leaves_bare_run_unchanged(self):
        night=original.night
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            source=root/'bare'
            destination=root/'original'
            for name in ('bbh3','bbh5'):
                registry=fixtures.NightTests().registry(root/name,name)
                folder,parent,parents=night.prepare(name,registry,source,'diffa',(16,5))
                night.generate_dataset(folder,parent,parents,fixtures.FakeBackend('diffa'))
                before={str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
                new,plan,plans,_=original.prepare(name,source,destination)
                self.assertNotEqual(plan['prompt'],parent['prompt'])
                self.assertEqual(plan['prompt'],plan['collection_prompt'])
                self.assertEqual(plan['recipe'],parent['recipe'])
                self.assertEqual(plan['fitting_ids'],parent['fitting_ids'])
                self.assertEqual(plan['records'],parent['records'])
                self.assertEqual(set(plans),{'angry','happy','sad'})
                self.assertEqual(plan['parent_cohort_sha256'],parent['cohort_sha256'])
                self.assertNotEqual(plan['cohort_sha256'],parent['cohort_sha256'])
                backend=fixtures.FakeBackend('diffa')
                night.generate_dataset(new,plan,plans,backend)
                night.generate_dataset(new,plan,plans,backend)
                self.assertEqual(len(backend.calls),64)
                self.assertEqual({s for _,s,_ in backend.calls},{'neutral','angry','happy','sad'})
                for e in night.EMOTIONS:
                    self.assertEqual(plans[e]['collection_ids'],parents[e]['collection_ids'])
                    self.assertTrue((new/e/'style-vectors.pt').exists())
                    job=night.steering_job(new,plan,plans[e])
                    old_job=night.steering_job(folder,parent,parents[e])
                    self.assertEqual(job['target']['records'],old_job['target']['records'])
                    self.assertEqual(job['target']['prompt'],original.original_prompt(name))
                    self.assertEqual(job['layer'],31)
                    self.assertEqual(job['strengths'],[0.,1.])
                self.assertEqual(before,{str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()})
                original.prepare(name,source,destination)
