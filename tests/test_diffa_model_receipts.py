import json
import os
from pathlib import Path
import tempfile
import unittest

from diffa_model_receipts import MODELS, cached_model_revisions, model_revisions


class ModelReceipts(unittest.TestCase):
    def test_source_cache_revisions_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory)
            for model in MODELS:
                folder=hub/('models--'+model.replace('/','--'))/'refs'
                folder.mkdir(parents=True)
                (folder/'main').write_text('a'*40)
            self.assertEqual(cached_model_revisions(hub),{m:'a'*40 for m in MODELS})
            next(hub.glob('models--*/refs/main')).write_text('main')
            with self.assertRaisesRegex(ValueError,'immutable'):
                cached_model_revisions(hub)

    def test_matching_receipts_and_conflicts(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                root=Path(directory); os.chdir(root)
                self.assertEqual(model_revisions(),{})
                (root/'outputs').mkdir()
                pins={model:'a'*40 for model in MODELS}
                (root/'outputs/diffa-model-revisions.json').write_text(json.dumps(dict(models=pins)))
                (root/'DIFFA_MIGRATION_MANIFEST.json').write_text(json.dumps(dict(source_environment=dict(hf_model_revisions=pins))))
                self.assertEqual(model_revisions(),pins)
                (root/'DIFFA_MIGRATION_MANIFEST.json').write_text(json.dumps(dict(source_environment=dict(hf_model_revisions={}))))
                self.assertEqual(model_revisions(),pins)
                (root/'DIFFA_MIGRATION_MANIFEST.json').write_text(json.dumps(dict(source_environment=dict(hf_model_revisions={MODELS[0]:'b'*40}))))
                with self.assertRaisesRegex(ValueError,'conflict'):
                    model_revisions()
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
