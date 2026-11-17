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


if __name__ == '__main__':
    unittest.main()
