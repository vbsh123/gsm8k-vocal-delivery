import json
from pathlib import Path
import tempfile
import unittest

from generate_missing_emotions import inventory


class MissingEmotionInventory(unittest.TestCase):
    def test_emotion_only_question_supplies_missing_neutral_and_keeps_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'happy.wav').write_bytes(b'x'*100)
            row = dict(provider='azure', voice='en-US-JennyNeural',
                       question_id='gsm8k_test_00001', question='How many?', answer='2',
                       style='happy', audio_path='happy.wav')
            (root/'plan.jsonl').write_text(json.dumps(row)+'\n')
            alias = dict(question_id=row['question_id'], question=row['question'], voice=row['voice'],
                         requested_style='happy')
            (root/'audio_inventory.jsonl').write_text(json.dumps(dict(paths=[str(root/'happy.wav')], aliases=[alias]))+'\n')
            bank, available = inventory(root)
            self.assertEqual(bank['gsm8k'][row['question_id']]['answer'], '2')
            self.assertNotIn((row['question_id'], 'neutral', row['question']), available)

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
