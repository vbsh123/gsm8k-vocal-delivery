import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import classify_cloud_smoke as run


class StyleFilterTests(unittest.TestCase):
    def test_angry_only_summary_preserves_other_saved_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'results-emotion2vec.jsonl'
            rows=[dict(requested_style=s,emotion=s) for s in ('neutral','angry','happy','sad')]
            content=''.join(json.dumps(r)+'\n' for r in rows);file.write_text(content)
            with patch.object(sys,'argv',['classify_cloud_smoke.py','--path',str(root),'--styles','angry','--summary-only']), \
                 redirect_stdout(io.StringIO()) as out:
                run.main()
            text=out.getvalue();self.assertIn('angry',text)
            for style in ('neutral','happy','sad'):self.assertNotIn(style,text)
            self.assertEqual(file.read_text(),content)

    def test_no_requested_style_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'plan.jsonl').write_text(json.dumps(dict(style='neutral'))+'\n')
            with patch.object(sys,'argv',['classify_cloud_smoke.py','--path',str(root),'--styles','angry']), \
                 redirect_stdout(io.StringIO()),patch('sys.stderr',io.StringIO()):
                with self.assertRaises(SystemExit):run.main()


if __name__=='__main__':unittest.main()
