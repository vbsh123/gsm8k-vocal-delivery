"""Pitch statistics, time alignment, and real-tone checks without inference."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from measure_audio_pitch import freeze_settings, summarize
from audit_speed_pitch import listening_page


class PitchReportTests(unittest.TestCase):
    def test_matched_pitch_uses_question_and_voice_and_ignores_unvoiced_clips(self):
        def row(q,voice,style,pitch):
            return dict(question_id=q,voice=voice,style=style,audio_file=f'{q}-{voice}-{style}',
                mean_f0_hz=pitch,median_f0_hz=pitch,pitch_span_semitones=2 if pitch else None,voiced_fraction=.6)
        rows=[row('q1','Jenny','neutral',200),row('q1','Jenny','angry',400),
              row('q2','Jenny','neutral',100),row('q2','Other','angry',400),row('q3','Jenny','angry',None)]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output=Path(directory)
            summarize(rows,output)
            result=json.loads((output/'summary.json').read_text())[0]
            self.assertEqual(result['pairs'],1)
            self.assertEqual(result['mean_f0_change_hz'],200)
            self.assertEqual(result['mean_median_pitch_change_semitones'],12)
            self.assertEqual(result['fraction_higher'],1)

    def test_changed_pitch_settings_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            freeze_settings(output,dict(fmin=50))
            with self.assertRaises(ValueError):
                freeze_settings(output,dict(fmin=100))
            self.assertEqual(json.loads((output/'pitch-settings.json').read_text()),dict(fmin=50))

    def test_listening_page_labels_reconstructions_and_embeds_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'audio').mkdir()
            (output/'audio/c__0.7x.wav').touch()
            row=dict(clip_id='c',rate=.7,duration_seconds=20,median_f0_hz=220,
                     waveform_source='reconstructed from recorded recipe')
            listening_page([dict(clip_id='c',question_id='question <1>')],[1,.7],[row],output,1)
            html=(output/'index.html').read_text()
            self.assertIn('question &lt;1&gt;',html)
            self.assertIn('src="audio/c__0.7x.wav"',html)
            self.assertIn('reconstructed from recorded recipe',html)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Needs NumPy')
class PitchAlignmentTests(unittest.TestCase):
    def test_time_alignment_uses_rate_and_drops_unvoiced_frames(self):
        import numpy as np
        from audit_speed_pitch import contour_difference
        base=np.array([100.,200.,300.,400.,500.,np.nan])
        fast=np.array([100.,300.,500.])
        result=contour_difference(base,fast,2,np)
        self.assertEqual(result['aligned_voiced_frames'],3)
        self.assertEqual(result['median_absolute_contour_shift_semitones'],0)
        result=contour_difference(base,fast*2,2,np)
        self.assertAlmostEqual(result['median_absolute_contour_shift_semitones'],12)
        result=contour_difference(base,np.full(3,np.nan),2,np)
        self.assertIsNone(result['median_absolute_contour_shift_semitones'])


@unittest.skipUnless(importlib.util.find_spec('librosa'),'Needs the audio libraries')
class RealPitchTests(unittest.TestCase):
    def test_time_stretch_preserves_a_known_tone_and_detects_pitch_shift(self):
        import librosa
        import numpy as np
        from measure_audio_pitch import measure_pitch
        t=np.arange(16000*2)/16000
        audio=(.5*np.sin(2*np.pi*220*t)+.2*np.sin(2*np.pi*440*t)).astype('float32')
        reference,_=measure_pitch(audio,librosa,np)
        self.assertLess(abs(reference['median_f0_hz']-220),5)
        for rate in (.7,1.5):
            changed=librosa.effects.time_stretch(audio,rate=rate)
            measured,_=measure_pitch(changed,librosa,np)
            semitones=12*math.log2(measured['median_f0_hz']/reference['median_f0_hz'])
            self.assertLess(abs(semitones),.3)
            self.assertEqual(len(changed),round(len(audio)/rate))
        shifted=librosa.effects.pitch_shift(audio,sr=16000,n_steps=3)
        measured,_=measure_pitch(shifted,librosa,np)
        self.assertAlmostEqual(12*math.log2(measured['median_f0_hz']/reference['median_f0_hz']),3,delta=.3)

    def test_silence_has_no_mean_pitch_instead_of_zero_hz(self):
        import librosa
        import numpy as np
        from measure_audio_pitch import measure_pitch
        stats,trace=measure_pitch(np.zeros(16000,dtype='float32'),librosa,np)
        self.assertIsNone(stats['mean_f0_hz'])
        self.assertIsNone(stats['median_f0_hz'])
        self.assertEqual(stats['voiced_frames'],0)
        self.assertTrue(np.isnan(trace).all())


if __name__=='__main__':
    unittest.main()
