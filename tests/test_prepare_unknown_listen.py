"""Check unknown-only sampling and portable audio copies without changing source data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from prepare_unknown_listen import main


class UnknownListenTests(unittest.TestCase):
    def test_copies_only_unknowns_and_makes_portable_archive_reproducibly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            records, predictions = [], []
            for i in range(6):
                (source / f"{i}.wav").write_bytes(f"audio{i}".encode())
                records.append({"clip_id": str(i), "audio_path": f"{i}.wav",
                                "text": "How many <apples> & pears?"})
                predictions.append({"clip_id": str(i), "binary_prediction":
                                    "unresolved" if i < 4 else "neutral"})
            for name, rows in [("manifest.jsonl", records),
                               ("results-emotion2vec-evaluated.jsonl", predictions)]:
                (source / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            selections = []
            for name in ("review1", "review2"):
                output = Path(directory) / name
                with patch.object(sys, "argv", ["test", "--path", str(source),
                                               "--output", str(output), "--count", "2"]):
                    with redirect_stdout(io.StringIO()):
                        main()
                selected = json.loads((output / "selection.json").read_text())["clip_ids"]
                selections.append(selected)
                self.assertEqual(len(selected), 2)
                self.assertTrue(set(selected) <= {"0", "1", "2", "3"})
                samples = [json.loads(l) for l in (output / "manifest.jsonl").read_text().splitlines()]
                for row in samples:
                    original = source / f"{row['clip_id']}.wav"
                    self.assertEqual((output / row["audio_path"]).read_bytes(), original.read_bytes())
                    self.assertFalse((output / row["audio_path"]).is_symlink())
                self.assertIn("&lt;apples&gt; &amp; pears", (output / "listen.html").read_text())
                with zipfile.ZipFile(str(output) + ".zip") as archive:
                    self.assertIn("listen.html", archive.namelist())
                    self.assertTrue(all(row["audio_path"] in archive.namelist() for row in samples))
            self.assertEqual(selections[0], selections[1])
            self.assertEqual(len(list(source.glob("*.wav"))), 6)


if __name__ == "__main__":
    unittest.main()
