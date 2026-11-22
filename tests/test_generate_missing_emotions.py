import json
from pathlib import Path
import tempfile
import unittest

from generate_missing_emotions import inventory


class MissingEmotionInventory(unittest.TestCase):
    def test_existing_audio_and_question_text_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'neutral.wav').write_bytes(b'x' * 100)
            (root / 'happy.wav').write_bytes(b'x' * 100)
            base = dict(provider='azure', voice='en-US-JennyNeural',
                        question_id='svamp_chal-1', question='How many?', answer='2')
            records = [dict(base, style='neutral', audio_path='neutral.wav'),
                       dict(base, style='happy', audio_path='happy.wav'),
                       dict(base, style='sad', audio_path='missing.wav')]
            (root / 'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            bank, available = inventory(root)
            self.assertEqual(len(bank['svamp']), 1)
            self.assertIn(('svamp_chal-1', 'happy', 'How many?'), available)
            self.assertNotIn(('svamp_chal-1', 'sad', 'How many?'), available)
            other = root / 'other'
            other.mkdir()
            (other / 'plan.jsonl').write_text(json.dumps(dict(records[0], question='Changed text'))+'\n')
            with self.assertRaisesRegex(ValueError, 'Differing neutral question texts'):
                inventory(root)


if __name__ == '__main__':
    unittest.main()
