import json
from pathlib import Path
import tempfile
import unittest

import shared_emotion_counts as run


class SharedCountsTests(unittest.TestCase):
    def test_three_labels_same_questions_neutral_unfiltered_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);clips=[];labels=[]
            for q in ['gsm8k_test_001','gsm8k_test_002','gsm8k_test_003']:
                for style in ('neutral','angry','happy','sad'):
                    key=q+style;path=root/(key+'.wav');path.write_bytes(b'audio')
                    clips.append(dict(clip_id=key,audio_sha256=key,paths=[str(path)],aliases=[
                        dict(question_id=q,requested_style=style,voice='en-US-JennyNeural')]*2))
                    if q.endswith('003') and style=='sad':continue
                    predicted='neutral' if q.endswith('002') and style=='sad' else style
                    if style=='neutral':predicted='surprised'
                    labels.append(dict(clip_id=key,audio_sha256=key,model=run.MODEL,
                                       processing_mode='whole_recording',classification_status='classified',emotion=predicted))
            (root/'audio_inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in clips))
            (root/'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            cohorts={'gsm8k':dict(test_ids=['gsm8k_test_001'],collection_ids=['gsm8k_test_002'])}
            r=run.count(root,cohorts)['gsm8k']
            self.assertEqual(r['shared_question_ids'],['gsm8k_test_001'])
            self.assertEqual(r['recognized']['happy'],3)
            self.assertEqual(r['shared_outside_original_test'],0)
            self.assertEqual(r['unclassified_emotional_clips'],1)
            # Missing neutral audio disqualifies a question even with all three labels.
            (root/'gsm8k_test_001neutral.wav').unlink()
            self.assertEqual(run.count(root,cohorts)['gsm8k']['shared_count'],0)


if __name__=='__main__':unittest.main()
