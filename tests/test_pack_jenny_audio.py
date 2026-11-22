import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


class JennyAudioArchive(unittest.TestCase):
    def test_only_validated_audio_and_hash_matched_labels_are_archived(self):
        script = Path(__file__).resolve().parents[1]/'pack_jenny_audio.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root/'outputs/jenny/audio.wav'
            audio.parent.mkdir(parents=True)
            data = b'original-audio-bytes'*20
            audio.write_bytes(data)
            (audio.parent/'other-voice.wav').write_bytes(b'not requested')
            key = hashlib.sha256(data).hexdigest()
            plan = audio.parent/'plan.jsonl'
            plan.write_text(json.dumps(dict(question_id='gsm8k_test_00001', style='sad', audio_path='audio.wav'))+'\n')
            clip = dict(audio_sha256=key, audio_file=str(audio), paths=[str(audio)],
                        aliases=[dict(source_config=str(plan))])
            check = root/'outputs/audio-transfer-check'
            check.mkdir()
            (check/'summary.json').write_text(json.dumps(dict(errors=[], valid_unique=1)))
            (check/'validated-audio.jsonl').write_text(json.dumps(clip)+'\n')
            label = dict(audio_sha256=key, clip_id=key, model='emotion2vec/emotion2vec_plus_large',
                         processing_mode='whole_recording', classification_status='classified', emotion='sad')
            (audio.parent/'results-emotion2vec.jsonl').write_text(json.dumps(label)+'\n')
            result = subprocess.run([sys.executable, str(script), '--destination', str(root/'download')],
                                    cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            archive_path = next((root/'download').glob('*.zip'))
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read('outputs/jenny/audio.wav'), data)
                self.assertNotIn('outputs/jenny/other-voice.wav', archive.namelist())
                self.assertIn('outputs/jenny/plan.jsonl', archive.namelist())
                saved = json.loads(archive.read('metadata/results-emotion2vec.jsonl'))
                self.assertEqual(saved['audio_sha256'], key)
            self.assertTrue(archive_path.with_suffix('.zip.sha256').is_file())


if __name__ == '__main__':
    unittest.main()
