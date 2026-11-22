import json
from pathlib import Path
import tempfile
import unittest

from report_original_happy_e2v import MODEL, angry_records, counts, read_labels


class OriginalHappyAgreement(unittest.TestCase):
    def test_exact_hash_and_unknown_labels_remain_separate(self):
        records = {'a': {'audio_sha256': 'used'}, 'b': {'audio_sha256': 'pending'}}
        result = counts(['a', 'b', 'c'], records, 'happy', {'used': 'sad', 'other-take': 'happy'})
        self.assertEqual((result['classified'], result['agreed'], result['pending'], result['missing_records']), (1, 0, 1, 1))
        self.assertEqual(result['agreement_pct'], 0.)

    def test_source_record_style_and_label_conflicts(self):
        record = dict(question_id='a', audio_sha256='hash')
        self.assertEqual(angry_records({'extraction': [dict(neutral=record, angry=record)]}), {'a': record})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results-emotion2vec.jsonl'
            base = dict(audio_sha256='hash', model=MODEL, processing_mode='whole_recording', classification_status='classified')
            path.write_text('\n'.join(json.dumps(dict(base, emotion=e)) for e in ('happy', 'angry')))
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                read_labels(Path(directory))


if __name__ == '__main__':
    unittest.main()
