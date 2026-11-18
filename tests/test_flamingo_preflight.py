from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import flamingo_disk_preflight as run


class DiskTests(unittest.TestCase):
    def test_fresh_and_resumed_environment_space(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root=Path(d);previous=Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ,{'HF_HUB_CACHE':str(root/'hub')}):
                    with patch.object(run.shutil,'disk_usage',return_value=SimpleNamespace(free=40*2**30)):
                        with self.assertRaisesRegex(RuntimeError,'No installation/download'):run.check()
                    env=root/'.venv-bbh-flamingo';env.mkdir()
                    with (env/'installed').open('wb') as stream:stream.truncate(8*2**30)
                    with patch.object(run.shutil,'disk_usage',return_value=SimpleNamespace(free=35*2**30)):
                        run.check()
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
