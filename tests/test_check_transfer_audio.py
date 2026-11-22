import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from check_transfer_audio import audit, validate


class TransferAudioIntegrity(unittest.TestCase):
    def test_dedup_missing_styles_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'audio.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b'\x01\x00' * 240)
            key = hashlib.sha256(path.read_bytes()).hexdigest()
            aliases = [dict(question_id='svamp_chal-1', voice='en-US-JennyNeural', requested_style=s)
                       for s in ('neutral', 'happy')]
            inventory = root / 'audio_inventory.jsonl'
            inventory.write_text(json.dumps(dict(audio_sha256=key, paths=[str(path)], aliases=aliases))+'\n')
            summary, clips = audit(inventory)
            self.assertEqual(summary['valid_unique'], 1)
            self.assertEqual(len(clips), 1)
            self.assertEqual(summary['datasets']['svamp']['styles']['happy'], 1)
            self.assertEqual(summary['datasets']['svamp']['missing']['sad'], ['svamp_chal-1'])
            self.assertAlmostEqual(validate(path, key)['seconds'], .01)
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                validate(path, key)
            truncated_key = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Truncated WAV'):
                validate(path, truncated_key)


if __name__ == '__main__':
    unittest.main()
