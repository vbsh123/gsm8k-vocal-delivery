import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from jenny_bundle_registry import Registry
from shared_emotion_counts import MODEL


class JennyBundleRegistry(unittest.TestCase):
    def test_staged_audio_hash_and_agreement_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'outputs/audio'
            source.mkdir(parents=True)
            metadata = root/'metadata'
            metadata.mkdir()
            records, plan, labels, mappings = [], [], [], []
            for index, style in enumerate(('neutral', 'angry', 'happy', 'sad'), 1):
                path = source/f'{style}.wav'
                with wave.open(str(path), 'wb') as audio:
                    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
                    audio.writeframes(bytes([index, 0])*100)
                key = hashlib.sha256(path.read_bytes()).hexdigest()
                old_path = f'/workspace/gsm8k-vocal-delivery/outputs/audio/{style}.wav'
                mappings.append(dict(source=old_path, member=f'outputs/audio/{style}.wav'))
                records.append(dict(audio_sha256=key, audio_file=old_path, paths=[old_path], aliases=[]))
                plan.append(dict(question_id='svamp_chal-1', provider='azure', voice='en-US-JennyNeural',
                                 style=style, question='How many?', answer='2', audio_path=path.name))
                if style != 'neutral':
                    labels.append(dict(audio_sha256=key, clip_id=key, model=MODEL,
                                       processing_mode='whole_recording', classification_status='classified', emotion=style))
            (root/'JENNY_AUDIO_MANIFEST.json').write_text(json.dumps(dict(files=mappings)))
            (metadata/'audio_inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (metadata/'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plan))
            registry = Registry(root)
            self.assertEqual(registry.neutral_ids('svamp'), ['svamp_chal-1'])
            self.assertIsNotNone(registry.choose('svamp', 'svamp_chal-1', 'angry', agreed=True))
            self.assertIsNone(registry.choose('svamp', 'svamp_chal-1', 'neutral', agreed=True))
            self.assertIsNone(registry.choose('svamp', 'svamp_chal-1', 'angry', question='Different text'))
            record = dict(question_id='svamp_chal-1', audio_sha256=records[0]['audio_sha256'])
            self.assertEqual(registry.relocate(record)['audio_file'], str(source/'neutral.wav'))
            with self.assertRaises(FileNotFoundError):
                registry.relocate(dict(record, audio_sha256='wrong'))


if __name__ == '__main__':
    unittest.main()
