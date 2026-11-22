import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from pack_diffa_migration import files_for, snapshot
from restore_diffa_bundle import restore


class DiffaMigration(unittest.TestCase):
    def test_bundle_keeps_checkpoint_and_dependencies_without_audio_or_pair_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'outputs/diffa-gsm-shared-three-zeroshot-hashes/angry'
            folder.mkdir(parents=True)
            old = root/'outputs/old'
            old.mkdir()
            (old/'style-vectors.pt').write_bytes(b'original vector')
            (folder/'run_config.json').write_text(json.dumps(dict(vector=str(old/'style-vectors.pt'))))
            (folder/'collection-stream.pt').write_bytes(b'resumable state')
            (folder/'results.jsonl').write_text('{"complete":true}\n{"partial":')
            (folder/'audio.wav').write_bytes(b'not included')
            (folder/'pairs').mkdir()
            (folder/'pairs/00001.pt').write_bytes(b'large activation cache')
            files, _ = files_for(root)
            self.assertIn((old/'style-vectors.pt').resolve(), files)
            self.assertIn((folder/'collection-stream.pt').resolve(), files)
            self.assertNotIn((folder/'pairs/00001.pt').resolve(), files)
            self.assertNotIn((folder/'audio.wav').resolve(), files)
            self.assertEqual(snapshot(folder/'results.jsonl'), b'{"complete":true}\n')

    def test_restore_reuses_identical_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'repo'
            root.mkdir()
            archive = Path(directory)/'bundle.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('JENNY_AUDIO_MANIFEST.json', '{}')
                z.writestr('outputs/a.wav', b'audio')
            restore(archive, root)
            restore(archive, root)
            (root/'outputs/a.wav').write_bytes(b'new local content')
            with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                restore(archive, root)
            self.assertEqual((root/'outputs/a.wav').read_bytes(), b'new local content')

    def test_restore_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'repo'
            root.mkdir()
            archive = Path(directory)/'bundle.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('JENNY_AUDIO_MANIFEST.json', '{}')
                z.writestr('../outside', b'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                restore(archive, root)


if __name__ == '__main__':
    unittest.main()
