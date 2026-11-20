import io,json,subprocess,zipfile
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import pack_phi_deliveries as export


class PhiDeliveryExportTests(unittest.TestCase):
    def fixture(self,repo,layer=17,count=2):
        folder=repo/'outputs/phi4-sweeps/svamp'/f'L{layer}-answer-a-1';folder.mkdir(parents=True)
        spec=dict(model=export.MODEL,revision='fixed')
        records=[dict(question_id=q,voice='Jenny',condition=s,style=s,audio_file='/audio.wav',audio_sha256=q+s,
            reference_answer='2') for q in ('q1','q2') for s in ('neutral','angry')]
        plan=dict(task='svamp',recipe=spec,prompt='TASK',seed=42,max_new_tokens=512,bank={q:{'answer':'2'} for q in ('q1','q2')},
            recordings=records,layer=layer,vector='/secret/vector.pt')
        config=folder/'run_config.json';config.write_text(json.dumps(plan))
        rows=[dict(r,prompt='TASK',seed=42,max_new_tokens=512,model=export.MODEL,revision='fixed',response='#### 2',response_tokens=3,
            strength=0,interventions=[]) for r in records[:count*2]]
        rows.append(dict(rows[0],condition='neutral-steered',strength=-1,interventions=[{'layer':layer}]))
        path=folder/'results.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return config,path

    def test_only_raw_baselines_exported_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);config,path=self.fixture(repo);before=path.read_bytes()
            target=export.export(repo)
            files=list(target.rglob('results.jsonl'));self.assertEqual(len(files),1)
            rows=[json.loads(s) for s in files[0].read_text().splitlines()]
            self.assertEqual(len(rows),4);self.assertEqual({r['condition'] for r in rows},{'neutral','angry'})
            self.assertTrue(all(not r['interventions'] for r in rows))
            plan=json.loads((files[0].parent/'run_config.json').read_text())
            self.assertNotIn('vector',plan);self.assertNotIn('layer',plan)
            self.assertEqual(path.read_bytes(),before)
            provenance=json.loads((files[0].parent/'PROVENANCE.json').read_text())
            for receipt,line in zip(provenance['exported_lines'],files[0].read_text().splitlines()):
                self.assertEqual(receipt['sha256'],export.sha(line.encode()))

    def test_identical_cohort_chooses_coverage_not_layer_or_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            repo=Path(tmp);self.fixture(repo,17,1);self.fixture(repo,30,2)
            target=export.export(repo);inventory=json.loads((target/'INVENTORY.json').read_text())
            self.assertEqual(len(inventory),1);self.assertEqual(inventory[0]['matched'],2)
            self.assertIn('L30',inventory[0]['source_config'])

    def test_archive_contains_no_steering_payloads(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);repo=root/'repo';repo.mkdir();config,path=self.fixture(repo)
            (config.parent/'style-vectors.pt').write_bytes(b'not exported')
            subprocess.run(['git','init','-q',str(repo)],check=True)
            (repo/'runner.py').write_text('pass\n')
            subprocess.run(['git','add','runner.py'],cwd=repo,check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','source'],cwd=repo,check=True)
            target=export.export(repo)
            from unittest.mock import patch
            with patch.dict(export.backup.GROUPS,{'phi-deliveries':(str(target.relative_to(repo/'outputs')),) }):
                archive=export.backup.package(repo,root/'downloads',samples=0,only=['phi-deliveries'])
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(any(name.endswith('.pt') for name in z.namelist()))
                self.assertFalse(any(name.startswith('outputs/phi4-sweeps/') for name in z.namelist()))
                result=next(name for name in z.namelist() if name.endswith('/results.jsonl'))
                self.assertNotIn('neutral-steered',z.read(result).decode())

    def test_hidden_intervention_under_neutral_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);config,path=self.fixture(repo)
            rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['strength']=-1
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'intervention'):export.candidate(config,repo/'outputs')

    def test_prompt_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);config,path=self.fixture(repo)
            rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['prompt']='DIFFERENT'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'mismatch'):export.candidate(config,repo/'outputs')


if __name__=='__main__':unittest.main()
