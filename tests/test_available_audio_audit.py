from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import audit_available_audio as run
from steer import read_rows


class AvailableAudioTests(unittest.TestCase):
    def fixture(self,root):
        repo=root/'repo';source=repo/'outputs/audio';source.mkdir(parents=True)
        sf.write(source/'neutral.wav',np.zeros(32000,dtype=np.float32),16000)
        shutil.copyfile(source/'neutral.wav',source/'angry.wav')
        tasks=[dict(question_id='q0',voice='Guy',style=s,audio_path=s+'.wav') for s in ('neutral','angry')]
        tasks.append(dict(question_id='q1',voice='Guy',style='sad',audio_path='missing.wav'))
        (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
        legacy=source/'results-emotion2vec.jsonl';legacy.write_text(json.dumps(dict(requested_style='angry',emotion='neutral'))+'\n')
        return repo,source,repo/'outputs/available-audio-emotions'

    def test_inventory_deduplicates_and_records_aliases_missing_audio_and_legacy_labels(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            repo,source,out=self.fixture(Path(d));clips=run.inventory(repo,out)
            self.assertEqual(len(clips),1)
            self.assertEqual({a['requested_style'] for a in clips[0]['aliases']},{'neutral','angry'})
            self.assertEqual(len(clips[0]['paths']),2)
            self.assertEqual(clips[0]['clip_id'],run.sha(source/'neutral.wav'))
            manifest=json.loads((out/'inventory_manifest.json').read_text())
            self.assertTrue(any('missing.wav' in str(i) for i in manifest['issues']))
            self.assertEqual(len(manifest['legacy_predictions']),1)
            # Future audio appears on a new snapshot; old hashes remain stable.
            sf.write(source/'missing.wav',np.ones(16000,dtype=np.float32)*.1,16000)
            again=run.inventory(repo,out);self.assertEqual(len(again),2)
            self.assertIn(clips[0]['clip_id'],{r['clip_id'] for r in again})

    @patch.dict('sys.modules', {'librosa': SimpleNamespace()})
    def test_whole_recording_full_scores_resume_and_model_error_receipts(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            repo,source,out=self.fixture(Path(d));run.inventory(repo,out);calls=[]
            def predict(audio):
                calls.append(len(audio));return dict(emotion='angry',labels=['neutral','angry'],scores=[.1,.9])
            adapter=SimpleNamespace(MODEL='test/model',__file__=run.__file__,load_predictor=lambda device:predict)
            original=run.importlib.import_module
            with patch.object(run.importlib,'import_module',side_effect=lambda name,*a,**k:adapter if name=='emotion2vec_agreement' else original(name,*a,**k)):
                run.classify(out,'emotion2vec','cpu');run.classify(out,'emotion2vec','cpu')
            self.assertEqual(calls,[32000])
            rows=read_rows(out/'results-emotion2vec.jsonl');self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['scores'],[.1,.9]);self.assertEqual(rows[0]['duration_seconds'],2.)
            def fail(audio):raise RuntimeError('simulated memory failure')
            broken=SimpleNamespace(MODEL='test/sensevoice',__file__=run.__file__,load_predictor=lambda device:fail)
            with patch.object(run.importlib,'import_module',side_effect=lambda name,*a,**k:broken if name=='sensevoice_agreement' else original(name,*a,**k)):
                run.classify(out,'sensevoice','cpu')
            error=read_rows(out/'results-sensevoice.jsonl')[0]
            self.assertEqual(error['classification_status'],'error');self.assertNotIn('emotion',error)
            self.assertIn('simulated memory',error['error'])
            self.assertEqual(json.loads((out/'summary.json').read_text())['models']['sensevoice']['errors'],1)


if __name__=='__main__':unittest.main()
