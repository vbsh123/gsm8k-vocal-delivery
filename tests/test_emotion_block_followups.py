from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import emotion_block_followups as follow
import tests.test_emotion_block_night as night_tests


class FollowupTests(unittest.TestCase):
    def make_cases(self,root,names):
        night=follow.night
        for name in names:
            registry=night_tests.NightTests().registry(root/'audio'/name,name)
            folder,plan,plans=night.prepare(name,registry,root/'outputs/emotion-block-night','diffa',(16,5))
            night.generate_dataset(folder,plan,plans,night_tests.FakeBackend('diffa'))
            (folder/'runtime.json').write_text(json.dumps(dict(model_revisions={m:str(i)*40 for i,m in enumerate(follow.MODELS,1)},block_collector_sha256='same-block-collector')))
            def generate(job,out,*args):
                with (out/'results.jsonl').open('a') as stream:
                    for record in job['target']['records']:
                        row=dict(record,response='#### 2',response_tokens=3,condition='L31-answer-a1',strength=1.,
                            prompt=job['target']['prompt'],seed=42,max_new_tokens=job['target']['max_new_tokens'],
                            steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                            vector_sha256=job['source']['vector_sha256'],interventions=[dict(layer=31,group='answer',strength=1.)])
                        stream.write(json.dumps(row)+'\n')
            with patch.object(night.winners,'generate_job',side_effect=generate), \
                 patch.object(night.vanilla.task,'report_steering'),patch.object(night,'steering_comparisons'):
                night.steer_dataset(folder,plan,plans,night_tests.FakeBackend('diffa'))
        return root/'outputs/emotion-block-night'

    def test_strengths_reuse_exact_zero_and_one_and_preserve_raw_vectors(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            primary=self.make_cases(root,follow.ROLES['old'])
            output=root/'followups'
            jobs=follow.prepare(primary,output,'old','strengths')
            self.assertEqual(len(jobs),6)
            self.assertEqual(follow.STRENGTHS,(-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.))
            for folder,job in jobs:
                self.assertEqual(job['source']['name'],job['target']['name'])
                self.assertEqual(job['strengths'],list(follow.STRENGTHS))
                done=follow.validate(job,folder)
                self.assertEqual(len(done),22)
                self.assertEqual({r['strength'] for r in done.values()},{0.,1.})
            before=[(folder/'results.jsonl').read_bytes() for folder,_ in jobs]
            follow.prepare(primary,output,'old','strengths')
            self.assertEqual(before,[(folder/'results.jsonl').read_bytes() for folder,_ in jobs])
            # Repair a torn last line without changing existing complete records.
            with (jobs[0][0]/'results.jsonl').open('ab') as stream:stream.write(b'{"partial":')
            follow.prepare(primary,output,'old','strengths')
            self.assertEqual((jobs[0][0]/'results.jsonl').read_bytes(),before[0])

    def test_all_transfer_cells_use_target_holdouts_recipe_and_only_zero_reuse(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            primary=self.make_cases(root,follow.DATASETS)
            cells=[]
            for role in follow.ROLES:
                jobs=follow.prepare(primary,root/'followups',role,'transfer')
                self.assertEqual(len(jobs),30)
                for folder,job in jobs:
                    self.assertNotEqual(job['source']['name'],job['target']['name'])
                    self.assertEqual(job['source']['emotion'],job['emotion'])
                    self.assertEqual(job['target']['emotion'],job['emotion'])
                    self.assertEqual(job['target']['prompt'],follow.night.catalog.prompt_for(job['target']['name']))
                    self.assertEqual(len(follow.validate(job,folder)),11)
                    self.assertTrue(all(r['strength']==0. for r in follow.validate(job,folder).values()))
                    self.assertFalse(set(job['source']['extraction_ids']) & {r['question_id'] for r in job['target']['records']})
                    cells.append((job['emotion'],job['source']['name'],job['target']['name']))
            self.assertEqual(len(set(cells)),90)
            self.assertFalse(any('phi'==source for _,source,_ in cells))

    def test_primary_archive_filters_caches_audio_and_followups_and_imports_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)/'source';root.mkdir()
            primary=self.make_cases(root,follow.ROLES['old'])
            cache=primary/'diffa/gsm8k/scoring/harness/large.pt'
            cache.parent.mkdir(parents=True);cache.write_bytes(b'exclude cache')
            (primary/'diffa/gsm8k/audio.wav').write_bytes(b'exclude audio')
            (root/'outputs/emotion-block-followups').mkdir()
            (root/'outputs/emotion-block-followups/result.json').write_text('{}')
            archive=root/'primary.zip'
            follow.archive_night(primary,'old',archive,root)
            with zipfile.ZipFile(archive) as stream:
                names=stream.namelist()
                self.assertIn('outputs/emotion-block-night/diffa/gsm8k/deliveries.jsonl',names)
                self.assertIn('outputs/emotion-block-night/diffa/multiarith/angry/style-vectors.pt',names)
                self.assertFalse(any('harness' in name or name.endswith('.wav') or 'followups' in name for name in names))
            target=Path(directory)/'target';target.mkdir()
            archive.rename(root/'emotion-night-old.zip')
            archive=root/'emotion-night-old.zip'
            follow.import_available(target/'outputs/emotion-block-night','new',target,root)
            follow.restore([archive],target,target/'outputs/emotion-block-night')
            record=target/'outputs/emotion-block-night/diffa/gsm8k/deliveries.jsonl'
            record.write_text('different existing file')
            with self.assertRaisesRegex(ValueError,'Existing primary'):
                follow.restore([archive],target,target/'outputs/emotion-block-night')
