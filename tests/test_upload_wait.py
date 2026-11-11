import hashlib
from pathlib import Path
import tempfile
import unittest

from run_azure_when_uploaded import archive_complete


class UploadTests(unittest.TestCase):
    def test_requires_exact_bytes_and_content_not_just_preallocated_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'upload.tar'
            content=b'complete upload'
            digest=hashlib.sha256(content).hexdigest()
            self.assertFalse(archive_complete(path,len(content),digest))
            path.write_bytes(content[:4])
            self.assertFalse(archive_complete(path,len(content),digest))
            path.write_bytes(b'0'*len(content))
            self.assertFalse(archive_complete(path,len(content),digest))
            path.write_bytes(content)
            self.assertTrue(archive_complete(path,len(content),digest))


if __name__=='__main__':
    unittest.main()
