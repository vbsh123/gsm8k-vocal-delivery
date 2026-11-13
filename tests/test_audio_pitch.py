"""A single invalid transform is excluded without losing valid prepared pairs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate_audio_pitch import CONDITIONS, PitchTransformError, prepare


class PitchPreparationTests(unittest.TestCase):
    def test_excludes_failed_pair_retains_cached_pair_and_resumes(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);audio=root/'audio';audio.mkdir()
            original=root/'original.wav';original.touch()
            def pair(q):
                return dict(pair_id=q,question_id=q,voice='Jenny',neutral=dict(audio_file=str(original)),
                            angry=dict(audio_file=str(original)))
            pairs=[pair('bad'),pair('good')]
            shifted={}
            for name in CONDITIONS[2:]:
                file=audio/f'good__{name}.wav';file.touch();shifted[name]=str(file)
                (audio/f'bad__{name}.wav').touch()
                (audio/f'bad__{name}.tmp.wav').touch()
            saved=dict(pair_id='good',status='prepared',shifted=shifted)
            (root/'pairs.jsonl').write_text(json.dumps(saved)+'\n')
            pitch=dict(median_f0_hz=200,voiced_frames=100,voiced_fraction=.8)
            modules={'librosa':SimpleNamespace(__version__='test'),
                     'numpy':SimpleNamespace(),
                     'soundfile':SimpleNamespace(write=lambda *a,**k:None),
                     'imageio_ffmpeg':SimpleNamespace(get_ffmpeg_exe=lambda:'ffmpeg')}
            args=SimpleNamespace(paths=[],limit=0,output=root)
            with patch.dict('sys.modules',modules),patch('evaluate_audio_pitch.select_pairs',return_value=pairs),\
                 patch('evaluate_audio_pitch.load_audio',return_value=[]),\
                 patch('evaluate_audio_pitch.measure_pitch',return_value=(pitch,None)),\
                 patch('evaluate_audio_pitch.subprocess.check_output',return_value='rubberband\n'),\
                 patch('evaluate_audio_pitch.shift_audio',side_effect=PitchTransformError('70ms drift')) as shift:
                usable,cache=prepare(args)
                self.assertEqual([p['pair_id'] for p in usable],['good'])
                self.assertEqual(cache['bad']['status'],'pitch_transform_unusable')
                self.assertEqual(cache['bad']['reason'],'70ms drift')
                self.assertTrue(all(Path(p).exists() for p in shifted.values()))
                self.assertFalse(list(audio.glob('bad__*')))
                usable,cache=prepare(args)
                self.assertEqual(shift.call_count,1)
                self.assertEqual([p['pair_id'] for p in usable],['good'])


if __name__=='__main__':unittest.main()
