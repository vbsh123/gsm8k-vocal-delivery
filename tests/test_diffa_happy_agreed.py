import json
from pathlib import Path
import tempfile
import unittest

import diffa_happy_agreed as run


class HappyAgreedTests(unittest.TestCase):
    def test_filter_uses_exact_audio_and_preserves_test_and_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            pairs = [dict(question_id=q, neutral={'audio_sha256': q+'n'}, emotion={'audio_sha256': q+'h'})
                     for q in ['keep', 'wrong-label', 'wrong-audio']]
            original = dict(name='gsm8k', collection_ids=[p['question_id'] for p in pairs], pairs=pairs,
                            test_ids=['test'], bank={q: dict(question=q, answer='2') for q in ['keep', 'wrong-label', 'wrong-audio']},
                            prompt='#### <number>', collection_prompt='Final answer: <number>')
            path = source/'run_config.json'; path.write_text(json.dumps(original)); before = path.read_bytes()
            inventory = {}
            for q in ['keep', 'wrong-label', 'wrong-audio']:
                for style, suffix in [('neutral', 'n'), ('happy', 'h')]:
                    if q == 'wrong-label' and style == 'happy': continue
                    sha = 'changed' if q == 'wrong-audio' and style == 'happy' else q+suffix
                    inventory[q, style] = [(dict(question=q), dict(audio_sha256=sha, reference_answer='2'), dict(predicted_emotion=style))]
            plan = run.prepare(source, root/'out', 'existing425', inventory)
            self.assertEqual(plan['collection_ids'], ['keep'])
            self.assertEqual(plan['test_ids'], ['test'])
            self.assertEqual(plan['prompt'], original['prompt'])
            self.assertEqual(plan['collection_prompt'], original['collection_prompt'])
            self.assertEqual(len(plan['classifier_receipts']), 1)
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_empty_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            (source/'run_config.json').write_text(json.dumps(dict(name='gsm8k', collection_ids=['q'], test_ids=['t'], pairs=[])))
            with self.assertRaisesRegex(ValueError, 'No neutral/happy-agreed'):
                run.prepare(source, root/'out', 'existing425', {})


if __name__ == '__main__':
    unittest.main()
