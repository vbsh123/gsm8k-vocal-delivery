"""Check dataset coverage/aggregation and held-out selection without loading models."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from classify_dataset_emotions import audio_index, archived_recordings, aggregate_predictions, chunk_ranges, report
from steer_azure_holdout import main as steer_main


class DatasetAuditTests(unittest.TestCase):
    def test_chunk_boundaries_rebalance_tiny_tails_without_padding_overlap_or_dropped_samples(self):
        for count in [400,401,480000,480001,480399,480400,960001]:
            ranges=chunk_ranges(count)
            self.assertEqual(ranges[0][0],0)
            self.assertEqual(ranges[-1][1],count)
            self.assertEqual(sum(end-start for start,end in ranges),count)
            self.assertTrue(all(400<=end-start<=480000 for start,end in ranges))
            self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))
        self.assertEqual(chunk_ranges(0),[])
        self.assertEqual(chunk_ranges(399),[])
        self.assertEqual(chunk_ranges(480001),[(0,479601),(479601,480001)])

    def test_qwen_launcher_uses_full_batches_and_reports_with_its_own_environment(self):
        from run_azure_followups import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            (root/'plan.jsonl').write_text('')
            argv=['test','--stage','qwen','--paths',str(root)]
            with patch.object(sys,'argv',argv),patch('run_azure_followups.run') as execute,patch('run_azure_followups.fcntl.flock'),patch('run_azure_followups.subprocess.check_output',return_value=''),patch('run_azure_followups.Path.open',return_value=io.StringIO()):
                main()
            commands=[c.args[0] for c in execute.call_args_list]
            self.assertEqual(commands[0][:6],['conda','run','--no-capture-output','-n','qwen-omni','python'])
            self.assertEqual(commands[0][commands[0].index('--num-questions')+1],'0')
            self.assertEqual(commands[0][-2:],['--seed','42'])
            self.assertTrue(all(c[:6]==commands[0][:6] for c in commands[1:]))
            self.assertTrue(all('--confidence-intervals' in c for c in commands[1:]))

    def test_multi_audio_metadata_deduplicates_files_preserving_all_questions(self):
        rows=[dict(id='q1',audio_path=['data/a.wav','data/b.wav'],category='speech'),
              dict(id='q2',audio_path=['data/a.wav'],category='speech')]
        index=audio_index(rows,'mmau-pro')
        self.assertEqual(len(index),2)
        self.assertEqual(index['data/a.wav']['question_ids'],['q1','q2'])
        self.assertEqual(index['data/b.wav']['categories'],['speech'])

    def test_zip_and_streaming_tar_use_metadata_paths_and_skip_completed(self):
        index=audio_index([dict(id='q',audio_id='./test-audios/a.wav',task='speech')],'mmau')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            zipped=root/'audio.zip'
            with zipfile.ZipFile(zipped,'w') as archive:
                archive.writestr('test-audios/a.wav',b'audio')
            tarred=root/'audio.tar.gz'
            with tarfile.open(tarred,'w:gz') as archive:
                info=tarfile.TarInfo('./test-audios/a.wav'); info.size=5
                archive.addfile(info,io.BytesIO(b'audio'))
            for path in (zipped,tarred):
                self.assertEqual(list(archived_recordings(path,index,set()))[0][1],b'audio')
                self.assertEqual(list(archived_recordings(path,index,set(index))),[])
                with self.assertRaises(ValueError):
                    list(archived_recordings(path,{'missing':{}},set()))
            for path,kind in [(zipped,'zip'),(tarred,'tar')]:
                with path.open('rb') as handle:
                    self.assertEqual(list(archived_recordings(handle,index,set(),kind))[0][1],b'audio')

    def test_speechr_nested_zip_unpack_preserves_existing_audio(self):
        from prepare_speechr_audio import unpack
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inner=io.BytesIO()
            with zipfile.ZipFile(inner,'w') as zipped:
                zipped.writestr('mini/mini_ori/q.wav',b'audio')
            outer=io.BytesIO()
            with zipfile.ZipFile(outer,'w') as zipped:
                zipped.writestr('audio_core/audio_core_mini.zip',inner.getvalue())
            with zipfile.ZipFile(outer) as zipped:
                self.assertEqual(unpack(zipped,root),1)
                self.assertEqual(unpack(zipped,root),0)
            self.assertEqual((root/'mini/mini_ori/q.wav').read_bytes(),b'audio')

    def test_duration_weighted_scores_and_votes_include_final_short_chunk(self):
        predictions=[dict(labels=['neutral','angry'],scores=[.9,.1]),
                     dict(labels=['neutral','angry'],scores=[.1,.9])]
        result=aggregate_predictions(predictions,[30,5],'emotion2vec')
        self.assertEqual(result['emotion'],'neutral')
        self.assertAlmostEqual(result['scores'][0],27.5/35)
        votes=aggregate_predictions([dict(emotion='neutral'),dict(emotion='angry')],[30,5],'sensevoice')
        self.assertEqual(votes['emotion'],'neutral')
        self.assertEqual(votes['duration_votes'],dict(neutral=30,angry=5))

    def test_report_counts_audio_files_not_repeated_question_references(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            rows=[dict(clip_id='a',question_ids=['q1','q2'],categories=['speech'],emotion='neutral'),
                  dict(clip_id='b',question_ids=['q1'],categories=['music'],emotion='happy')]
            (root/'results-emotion2vec.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary=report(root,'emotion2vec')
            self.assertEqual(summary['unique_recordings'],2)
            self.assertEqual(summary['categories']['speech'],dict(neutral=1))

    def test_holdout_uses_neutral_audio_without_outcome_or_classifier_filter(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); data=root/'data'; data.mkdir(); vector=root/'vector.pt'; vector.touch()
            tasks=[]
            for i,style in enumerate(['angry','neutral','neutral','neutral']):
                tasks.append(dict(provider='azure',question_id=f'q{i}',style=style,voice='Jenny',answer='reason #### 18',audio_path=f'{i}.wav'))
                (data/f'{i}.wav').touch()
            (data/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
            argv=['test','--path',str(data),'--vectors',str(vector),'--output',str(root/'run'),
                  '--num-examples','2','--prepare-only']
            with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'soundfile':SimpleNamespace(info=lambda path:SimpleNamespace(duration=10))}),patch('steer_azure_holdout.subprocess.run') as execute:
                steer_main()
            plan=json.loads((root/'run/selected-plan.json').read_text())
            self.assertEqual([r['question_id'] for r in plan['recordings']],['q1','q2'])
            self.assertEqual(plan['recordings'][0]['answer'],'18')
            self.assertEqual(plan['conditions'][1]['interventions'],[dict(layer=12,group='acoustic',strength=1)])
            self.assertIn('--prepare-only',execute.call_args.args[0])


@unittest.skipUnless(importlib.util.find_spec('soundfile') and importlib.util.find_spec('librosa'),'Needs audio libraries')
class RealDatasetAuditTests(unittest.TestCase):
    def test_tiny_tail_is_rebalanced_and_short_whole_clip_is_reported_not_classified(self):
        import numpy as np
        import soundfile as sf
        from classify_dataset_emotions import main
        calls=[]
        class Adapter:
            MODEL='test-model'
            def load_predictor(device):
                def predict(audio):
                    if len(audio)<400:
                        raise RuntimeError('Calculated padded input size too small')
                    calls.append(len(audio))
                    return dict(emotion='neutral')
                return predict
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); data=root/'audio'; data.mkdir()
            sf.write(data/'long.wav',np.zeros(480001,dtype='float32'),16000)
            sf.write(data/'short.wav',np.zeros(100,dtype='float32'),16000)
            argv=['test','--dataset','speechr','--audio-root',str(data),'--output',str(root/'out'),
                  '--model','sensevoice','--device','cpu']
            with patch.object(sys,'argv',argv),patch('classify_dataset_emotions.importlib.import_module',return_value=Adapter),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(HfApi=lambda:None)}):
                main(); main()
            self.assertEqual(calls,[479601,400])
            rows=[json.loads(line) for line in (root/'out/results-sensevoice.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['classification_status'],'insufficient_audio')
            summary=json.loads((root/'out/summary-sensevoice.json').read_text())
            self.assertEqual(summary['classified_recordings'],1)
            self.assertEqual(summary['unclassified_recordings'],1)

    def test_cloud_classifier_resume_preserves_predictions_and_does_not_load_again(self):
        from classify_cloud_smoke import main
        import numpy as np
        import soundfile as sf
        calls=[]
        class Adapter:
            MODEL='test-model'
            def load_predictor(device):
                calls.append('load')
                return lambda audio:dict(emotion='neutral',scores=[1.])
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            sf.write(root/'q.wav',np.zeros(16000,dtype='float32'),16000)
            (root/'plan.jsonl').write_text(json.dumps(dict(provider='azure',style='neutral',voice='Jenny',audio_path='q.wav',question_id='q'))+'\n')
            argv=['test','--path',str(root),'--model','emotion2vec','--device','cpu']
            with patch.object(sys,'argv',argv),patch('classify_cloud_smoke.importlib.import_module',return_value=Adapter):
                main()
                before=(root/'results-emotion2vec.jsonl').read_text()
                main()
            self.assertEqual(before,(root/'results-emotion2vec.jsonl').read_text())
            self.assertEqual(calls,['load'])

    def test_long_recording_uses_all_chunks_and_resumes_without_reloading_model(self):
        import numpy as np
        import soundfile as sf
        from classify_dataset_emotions import main
        calls=[]
        class Adapter:
            MODEL='test-model'
            def load_predictor(device):
                calls.append('load')
                def predict(audio):
                    calls.append(len(audio))
                    return dict(emotion='neutral')
                return predict
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); data=root/'audio'; data.mkdir()
            sf.write(data/'q.wav',np.zeros(65*16000,dtype='float32'),16000)
            argv=['test','--dataset','speechr','--audio-root',str(data),'--output',str(root/'out'),
                  '--model','sensevoice','--device','cpu']
            with patch.object(sys,'argv',argv),patch('classify_dataset_emotions.importlib.import_module',return_value=Adapter),patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(HfApi=lambda:None)}):
                main(); main()
            self.assertEqual(calls,['load',480000,480000,80000])
            rows=(root/'out/results-sensevoice.jsonl').read_text().splitlines()
            self.assertEqual(len(rows),1)
            self.assertEqual(json.loads(rows[0])['duration_seconds'],65)


if __name__=='__main__':
    unittest.main()
