import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from classify_shared_emotions import prepare
from shared_emotion_counts import MODEL, count


class SharedEmotionClassification(unittest.TestCase):
    def test_hash_reuse_and_same_question_intersection_without_neutral_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = []
            labels = []
            for index in (1, 2, 3):
                for style in ('neutral', 'angry', 'happy', 'sad'):
                    filename = f'{index}-{style}.wav'
                    data = (f'{index}-{style}' * 20).encode()
                    (root / filename).write_bytes(data)
                    plan.append(dict(provider='azure', voice='en-US-JennyNeural',
                                     question_id=f'svamp_chal-{index}', question=f'Question {index}',
                                     style=style, audio_path=filename))
                    if style == 'neutral' or (index == 3 and style == 'sad'):
                        continue
                    key = hashlib.sha256(data).hexdigest()
                    labels.append(dict(clip_id=key, audio_sha256=key, model=MODEL,
                                       processing_mode='whole_recording', classification_status='classified',
                                       emotion='neutral' if index == 2 and style == 'sad' else style))
            (root / 'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plan))
            (root / 'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            worker, pending = prepare(root, root / 'shared-three-emotions')
            self.assertEqual(pending, 1)
            inventory = [json.loads(s) for s in (worker / 'audio_inventory.jsonl').read_text().splitlines()]
            self.assertEqual(len(inventory), 9)
            self.assertTrue(all(a['requested_style'] != 'neutral' for r in inventory for a in r['aliases']))
            summary = count(root)['svamp']
            self.assertEqual(summary['shared_question_ids'], ['svamp_chal-1'])
            self.assertEqual(summary['unclassified_emotional_clips'], 1)


if __name__ == '__main__':
    unittest.main()
