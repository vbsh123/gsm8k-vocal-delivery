from contextlib import redirect_stdout
from http.client import IncompleteRead
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from urllib.error import HTTPError

from cloud_tts_smoke import main,request_audio,TransientTTSError


class TTSRetryTests(unittest.TestCase):
    def test_incomplete_response_is_retried_without_returning_partial_bytes(self):
        response=MagicMock()
        response.__enter__.return_value.read.side_effect=[IncompleteRead(b'partial'),b'complete']
        with patch('cloud_tts_smoke.urlopen',return_value=response) as call,\
             patch('cloud_tts_smoke.time.sleep'),redirect_stdout(io.StringIO()):
            self.assertEqual(request_audio('https://example.test',b'body',{}),b'complete')
            self.assertEqual(call.call_count,2)

    def test_auth_error_is_not_retried(self):
        error=HTTPError('https://example.test',401,'unauthorized',{},io.BytesIO(b'bad credentials'))
        with patch('cloud_tts_smoke.urlopen',side_effect=error) as call:
            with self.assertRaisesRegex(RuntimeError,'401'):request_audio('https://example.test',b'body',{})
            self.assertEqual(call.call_count,1)

    def test_resume_preserves_saved_clip_and_revisits_failed_clip_after_others(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d)
            argv=['test','--provider','azure','--styles','neutral','angry','happy','--output',d,'--resume','--retry-passes','2']
            with patch('sys.argv',argv+['--prepare-only']):main()
            saved=root/'azure-neutral.wav';saved.write_bytes(b'already complete')
            with patch('sys.argv',argv),\
                 patch.dict('os.environ',AZURE_SPEECH_KEY='test',AZURE_SPEECH_REGION='test'),\
                 patch('cloud_tts_smoke.time.sleep'),\
                 patch('cloud_tts_smoke.request_audio',side_effect=[TransientTTSError('interrupted'),b'happy',b'angry']) as call:
                main()
            self.assertEqual(saved.read_bytes(),b'already complete')
            self.assertEqual((root/'azure-angry.wav').read_bytes(),b'angry')
            self.assertEqual((root/'azure-happy.wav').read_bytes(),b'happy')
            self.assertEqual(call.call_count,3)
            self.assertFalse(list(root.glob('*.tmp')))


if __name__=='__main__':unittest.main()
