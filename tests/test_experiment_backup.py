from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import pack_experiment_backup as run


class EvidenceBackupTests(unittest.TestCase):
    def test_emotions_archive_includes_complete_new_and_historical_labels(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d);repo,source,_,_=self.repo(root)
            audit=repo/'outputs/available-audio-emotions';audit.mkdir()
            legacy=source/'results-sensevoice.jsonl';legacy.write_text('{"emotion":"neutral"}\n')
            (audit/'inventory_manifest.json').write_text(json.dumps(dict(legacy_predictions=[dict(path=str(legacy))])))
            for model in ('emotion2vec','sensevoice'):
                (audit/f'results-{model}.jsonl').write_text(json.dumps(dict(audio_sha256='abc',emotion='angry'))+'\n')
            path=run.package(repo,root/'downloads',samples=0,only=['emotions'])
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist()
                self.assertIn('outputs/available-audio-emotions/results-emotion2vec.jsonl',names)
                self.assertIn('outputs/available-audio-emotions/results-sensevoice.jsonl',names)
                self.assertIn(str(legacy.relative_to(repo)),names)

    def test_transfer_and_new_voice_backups_include_actual_answers_and_counts(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d);repo,source,_,_=self.repo(root)
            transfer=repo/'outputs/winning-transfer/gsm8k-to-svamp';transfer.mkdir(parents=True)
            row=dict(question_id='q0',voice='Guy',condition='neutral',response='#### 2',audio_file=str(source/'q0-neutral.wav'))
            (transfer/'run_config.json').write_text(json.dumps(dict(kind='transfer',strengths=[0,1],target=dict(records=[row]))))
            (transfer/'results.jsonl').write_text(json.dumps(row)+'\n')
            voice=repo/'outputs/multiarith290-guy-delivery';voice.mkdir()
            (voice/'run_config.json').write_text(json.dumps(dict(recordings=[row])))
            (voice/'results.jsonl').write_text(json.dumps(row)+'\n')
            path=run.package(repo,root/'downloads',samples=0,only=['transfer','voice'])
            with zipfile.ZipFile(path) as archive:
                manifest=json.loads(archive.read('MANIFEST.json'))
                counts={r['experiment']:r for r in manifest['runs']}
                self.assertEqual(counts['DIFFA transfer gsm8k-to-svamp']['planned_answers'],2)
                self.assertEqual(counts['DIFFA MultiArith Guy delivery']['saved_unique_answers'],1)
                self.assertIn('outputs/winning-transfer/gsm8k-to-svamp/results.jsonl',archive.namelist())

    def test_phi_all_layer_worker_answers_are_packaged_and_counted(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d);repo,_,_,_=self.repo(root)
            folder=repo/'outputs/phi4-gsm400-all-layers-test869-minus1';folder.mkdir()
            plan=dict(protocol='phi4-test',evaluation_kind='all-layer-diagnostic',recordings=[],layers=list(range(32)),groups=['answer','audio'])
            (folder/'run_config.json').write_text(json.dumps(plan))
            (folder/'challenge.json').write_text(json.dumps(dict(cases=[dict(question_id='q0'),dict(question_id='q1')])))
            (folder/'style-vectors.pt').write_bytes(b'vectors')
            for i in range(2):
                (folder/f'results-worker{i}.jsonl').write_text(json.dumps(dict(question_id=f'q{i}',voice='Jenny',condition='neutral-L0-answer-a-1',response='2'))+'\n')
            path=run.package(repo,root/'downloads',samples=0,only=['phi'])
            with zipfile.ZipFile(path) as archive:
                m=json.loads(archive.read('MANIFEST.json'))
                row=next(r for r in m['runs'] if r['experiment']=='Phi all-layer recovery diagnostic')
                self.assertTrue(row['answers_present']);self.assertEqual(row['saved_unique_answers'],2)
                self.assertEqual(row['planned_answers'],128)
                self.assertIn('outputs/phi4-gsm400-all-layers-test869-minus1/results-worker1.jsonl',archive.namelist())

    def test_strengths_include_raw_answers_external_vector_and_nested_audio(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d); repo, source, _, _ = self.repo(root)
            folder=repo/'outputs/winning-strengths/gsm8k'; folder.mkdir(parents=True)
            external=repo/'outputs/other-vector';external.mkdir();vector=external/'direction.pt';vector.write_bytes(b'vector')
            row=dict(question_id='q0', voice='Jenny', style='neutral', audio_file=str(source/'q0-neutral.wav'))
            plan=dict(kind='strengths',source=dict(vector=str(vector)),target=dict(records=[row]),
                      strengths=[-2,-1.5,-1,-.5,0,.5,1,1.5,2])
            (folder/'run_config.json').write_text(json.dumps(plan))
            (folder/'results.jsonl').write_text(json.dumps(dict(row,condition='neutral',response='#### 2'))+'\n')
            path=run.package(repo,root/'downloads',samples=1,only=['strengths'])
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist();m=json.loads(archive.read('MANIFEST.json'))
                self.assertIn('outputs/winning-strengths/gsm8k/results.jsonl',names)
                self.assertTrue(any(n.startswith('referenced-vectors/') for n in names))
                self.assertEqual(len(m['audio_samples']),1)
                status=next(r for r in m['runs'] if r['experiment']=='DIFFA strengths gsm8k')
                self.assertEqual(status['planned_answers'],9)
                self.assertEqual(status['saved_unique_answers'],1)
                self.assertTrue(status['required_vector_present'])

    def test_phi_focus_preserves_small_collection_means_and_ignores_unrelated_runs(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d);repo,source,_,_=self.repo(root)
            folder=repo/'outputs/phi4-gsm-reverse400-400-L17';folder.mkdir()
            (folder/'pairs').mkdir();(folder/'pairs/0000.pt').write_bytes(b'raw collection responses and means')
            (folder/'style-vectors.pt').write_bytes(b'collected direction')
            (folder/'run_config.json').write_text(json.dumps(dict(protocol='phi4-gsm-test',collection=[],test=[dict(question_id='q0')])))
            (folder/'results.jsonl').write_text(json.dumps(dict(question_id='q0',voice='Jenny',condition='neutral',response='#### 2'))+'\n')
            shutil.copytree(source,repo/'outputs/azure-kimi-gsm8k1269')
            path=run.package(repo,root/'downloads',samples=1,only=['phi'])
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist();m=json.loads(archive.read('MANIFEST.json'))
                self.assertIn('outputs/phi4-gsm-reverse400-400-L17/pairs/0000.pt',names)
                self.assertIn('outputs/phi4-gsm-reverse400-400-L17/style-vectors.pt',names)
                self.assertFalse(any(n.startswith('outputs/bbh-logical250-angry-mean/') for n in names))
                self.assertEqual(len(m['runs']),1)
                self.assertEqual(m['runs'][0]['saved_unique_answers'],1)

    def test_ifeval_relative_audio_uses_matching_source_plan_and_hash(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); repo, old_source, _, _ = self.repo(root)
            source = repo/'outputs/azure-spoken-ifeval'; shutil.copytree(old_source, source)
            rows = [json.loads(s) for s in (source/'plan.jsonl').read_text().splitlines()]
            evaluation = repo/'outputs/spoken-ifeval-diffa2'; evaluation.mkdir()
            tasks = [dict(row, audio_sha256=hashlib.sha256((source/row['audio_path']).read_bytes()).hexdigest()) for row in rows]
            (evaluation/'run_config.json').write_text(json.dumps(dict(recordings=tasks)))
            path = run.package(repo, root/'downloads', samples=1, only=['ifeval'])
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('MANIFEST.json'))
                self.assertFalse(any('sample audio missing' in x['error'] for x in manifest['issues']))
                self.assertEqual(len(manifest['audio_samples']), 2)
                self.assertTrue(all('/azure-spoken-ifeval/' in a['member'] for a in manifest['audio_samples']))
            # A similarly named file with different audio bytes cannot satisfy the receipt.
            row = dict(tasks[0], audio_sha256='0'*64)
            index = run.source_audio_index([source/'plan.jsonl'])
            chosen = run.audio_path(row, evaluation/'run_config.json', repo, index)
            self.assertFalse(chosen.is_file())

    def repo(self, root):
        repo = root/'repo'; repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        (repo/'runner.py').write_text('print("experiment")\n')
        subprocess.run(['git', 'add', 'runner.py'], cwd=repo, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'source'], cwd=repo, check=True)
        outputs = repo/'outputs'; outputs.mkdir()
        source = outputs/'azure-test'; source.mkdir()
        rows = []
        for i in range(3):
            for style in ('neutral', 'angry'):
                name = f'q{i}-{style}.wav'
                (source/name).write_bytes(b'RIFF'+bytes([i])*40+style.encode())
                rows.append(dict(provider='azure', question_id=f'q{i}', voice='Jenny', style=style, audio_path=name))
        (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result = outputs/'bbh-logical250-angry-mean'; result.mkdir()
        partial = b'{"question_id":"q0","voice":"Jenny","condition":"neutral"}\n{"partial":'
        (result/'results.jsonl').write_bytes(partial)
        (result/'split.json').write_text(json.dumps(dict(test=[dict(question_id='q0')], extraction=[])))
        (result/'style-vectors.pt').write_bytes(b'collected vector')
        (result/'pairs').mkdir(); (result/'pairs/00000.pt').write_bytes(b'large intermediate activation cache')
        (outputs/'hf-diffa').mkdir(); (outputs/'hf-diffa/config.json').write_text('{"model_cache":true}')
        return repo, source, result, partial

    def test_valid_readonly_snapshot_vector_code_paired_samples_and_partial_answer_tail(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); repo, source, result, partial = self.repo(root)
            path = run.package(repo, root/'downloads', samples=1)
            self.assertNotIn(repo/'outputs', path.parents)
            self.assertEqual((result/'results.jsonl').read_bytes(), partial)
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist()); manifest = json.loads(archive.read('MANIFEST.json'))
                self.assertIn('code/runner.py', members)
                self.assertIn('outputs/bbh-logical250-angry-mean/style-vectors.pt', members)
                self.assertNotIn('outputs/bbh-logical250-angry-mean/pairs/00000.pt', members)
                self.assertNotIn('outputs/hf-diffa/config.json', members)
                self.assertEqual({a['style'] for a in manifest['audio_samples']}, {'neutral', 'angry'})
                self.assertEqual(len(manifest['audio_samples']), 2)
                saved = archive.read('outputs/bbh-logical250-angry-mean/results.jsonl')
                self.assertEqual(json.loads(saved)['question_id'], 'q0')
                for entry in manifest['files']:
                    self.assertEqual(hashlib.sha256(archive.read(entry['member'])).hexdigest(), entry['sha256'])
                status = next(r for r in manifest['runs'] if r['experiment'] == 'BBH three steering')
                self.assertEqual((status['saved_unique_answers'], status['planned_answers']), (1, 3))
            self.assertTrue(path.with_suffix('.zip.sha256').exists())

    def test_full_audio_dereferences_aliases_and_restores_every_logical_path(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); repo, source, _, _ = self.repo(root)
            alias = repo/'outputs/azure-alias'; alias.mkdir()
            (alias/'reused.wav').symlink_to(source/'q0-neutral.wav')
            (alias/'plan.jsonl').write_text(json.dumps(dict(provider='azure', question_id='q0',
                voice='Jenny', style='neutral', audio_path='reused.wav'))+'\n')
            path = run.package(repo, root/'downloads', samples=0, full_audio=True)
            full = path.with_name(path.stem+'-full-tts-audio.zip')
            with zipfile.ZipFile(full) as archive:
                self.assertEqual(len(archive.namelist()), 7)
                self.assertEqual(archive.read('outputs/azure-alias/reused.wav'), (source/'q0-neutral.wav').read_bytes())
                self.assertEqual(archive.read('outputs/azure-test/q0-neutral.wav'), (source/'q0-neutral.wav').read_bytes())
            self.assertTrue((alias/'reused.wav').is_symlink())

    def test_refuses_archive_inside_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            repo, _, _, _ = self.repo(Path(d))
            with self.assertRaisesRegex(ValueError, 'outside outputs'):
                run.package(repo, repo/'outputs/backups')

    def test_focused_backup_includes_new_run_audio_vector_and_excludes_old_results(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); repo, source, result, _ = self.repo(root)
            shutil.copytree(result, repo/'outputs/multiarith580-angry-mean')
            shutil.copytree(source, repo/'outputs/azure-multiarith580')
            path = run.package(repo, root/'downloads', samples=1, only=['multiarith'])
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist(); manifest = json.loads(archive.read('MANIFEST.json'))
                self.assertIn('outputs/multiarith580-angry-mean/results.jsonl', names)
                self.assertIn('outputs/multiarith580-angry-mean/style-vectors.pt', names)
                self.assertIn('outputs/azure-multiarith580/plan.jsonl', names)
                self.assertIn('code/runner.py', names)
                self.assertFalse(any(n.startswith('outputs/bbh-logical250-angry-mean/') for n in names))
                self.assertFalse(any(n.startswith('outputs/azure-test/') for n in names))
                self.assertEqual({a['style'] for a in manifest['audio_samples']}, {'neutral', 'angry'})
                self.assertEqual([r['experiment'] for r in manifest['runs']], ['MultiArith steering'])

    def test_step_vector_backup_keeps_per_pair_means_only_when_requested(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);repo,_,_,_=self.repo(root)
            folder=repo/'outputs/step-gsm400-vectors';(folder/'pairs').mkdir(parents=True)
            (folder/'pairs/00000.pt').write_bytes(b'x'*(3*1024**2))
            (folder/'style-vectors.pt').write_bytes(b'raw mean')
            (folder/'run_config.json').write_text(json.dumps(dict(recordings=[])))
            path=run.package(repo,root/'downloads',samples=0,only=['step-vectors'])
            with zipfile.ZipFile(path) as z:
                self.assertIn('outputs/step-gsm400-vectors/pairs/00000.pt',z.namelist())
                self.assertIn('outputs/step-gsm400-vectors/style-vectors.pt',z.namelist())
            path=run.package(repo,root/'downloads',samples=0,only=['step'])
            with zipfile.ZipFile(path) as z:
                self.assertNotIn('outputs/step-gsm400-vectors/pairs/00000.pt',z.namelist())
                self.assertIn('outputs/step-gsm400-vectors/style-vectors.pt',z.namelist())


if __name__ == '__main__':
    unittest.main()
