from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import wave

from run_commonsense_models import remove_weights,completed,main,pin_diffa_cache,restore_diffa_cache


class CommonsenseQueueTests(unittest.TestCase):
    def test_pins_and_restores_original_model_revisions_and_file_selection(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);pins=home/'pins.json'
            for repo in ('GSAI-ML--LLaDA-8B-Instruct','openai--whisper-large-v3','zhoujiaming777--DIFFA-2'):
                folder=home/'hub'/('models--'+repo);(folder/'refs').mkdir(parents=True)
                (folder/'refs/main').write_text('original-sha')
                snapshot=folder/'snapshots/original-sha';snapshot.mkdir(parents=True)
                (snapshot/'config.json').write_text('{}');(snapshot/'weights.bin').write_text('weights')
            pin_diffa_cache(home,pins)
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(snapshot_download=lambda *a,**k:None)}):
                restore_diffa_cache(home,pins)
            values=json.loads(pins.read_text())
            self.assertTrue(all(r['revision']=='original-sha' for r in values.values()))
            self.assertTrue(all(set(r['files'])=={'config.json','weights.bin'} for r in values.values()))

    def test_queue_runs_qwen_then_kimi_then_diffa_steering(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'source';source.mkdir();(root/'outputs').mkdir();tasks=[]
            for style in ('neutral','angry','happy','sad'):
                with wave.open(str(source/(style+'.wav')),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*160)
                tasks.append(dict(provider='azure',style=style,audio_path=style+'.wav'))
            (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
            events=[];done=set()
            def run(command,**kwargs):
                events.append(command)
                if 'evaluate_qwen_omni.py' in command:done.add('qwen')
                if 'evaluate_kimi_audio.py' in command:done.add('kimi')
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch('sys.argv',['test','--source',str(source),'--output',str(root/'out'),'--release-diffa-cache','--release-model-cache']),\
                     patch.dict('os.environ',HF_HOME=str(root/'cache')),\
                     patch('run_commonsense_models.subprocess.run',side_effect=run),\
                     patch('run_commonsense_models.subprocess.check_output',return_value=''),\
                     patch('run_commonsense_models.shutil.disk_usage',return_value=SimpleNamespace(free=40*2**30)),\
                     patch('run_commonsense_models.pin_diffa_cache'),patch('run_commonsense_models.remove_weights'),\
                     patch('run_commonsense_models.cached',return_value=False),\
                     patch('run_commonsense_models.completed',side_effect=lambda source,label,expected:label in done):
                    main()
            finally:os.chdir(previous)
            q=next(i for i,c in enumerate(events) if 'evaluate_qwen_omni.py' in c)
            k=next(i for i,c in enumerate(events) if 'evaluate_kimi_audio.py' in c)
            restore=next(i for i,c in enumerate(events) if '--restore-diffa-only' in c)
            steering=next(i for i,c in enumerate(events) if 'steer_commonsense_mean.py' in c)
            self.assertLess(q,k);self.assertLess(k,restore);self.assertLess(restore,steering)

    def test_weight_release_preserves_other_models_and_saved_results(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);cache=root/'cache';repo=cache/'hub/models--GSAI-ML--LLaDA-8B-Instruct';repo.mkdir(parents=True)
            (repo/'weights').write_text('downloadable')
            whisper=cache/'hub/models--openai--whisper-large-v3';whisper.mkdir();(whisper/'weights').touch()
            data=root/'results.jsonl';data.write_text('preserve')
            alias=root/'alias';(alias/'hub').mkdir(parents=True)
            (alias/'hub/models--GSAI-ML--LLaDA-8B-Instruct').symlink_to(repo,target_is_directory=True)
            with patch('run_commonsense_models.cache_roots',return_value=[alias,cache]):remove_weights('GSAI-ML--LLaDA-8B-Instruct')
            self.assertFalse(repo.exists());self.assertFalse((alias/'hub/models--GSAI-ML--LLaDA-8B-Instruct').is_symlink())
            self.assertTrue((whisper/'weights').exists());self.assertEqual(data.read_text(),'preserve')

    def test_prepare_checks_audio_without_loading_models_or_releasing_weights(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);tasks=[]
            for style in ('neutral','angry','happy','sad'):
                name=style+'.wav'
                with wave.open(str(root/name),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*160)
                tasks.append(dict(provider='azure',style=style,audio_path=name))
            (root/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
            argv=['test','--source',str(root),'--output',str(root/'out'),'--skip-steering','--prepare-only','--release-diffa-cache']
            with patch('sys.argv',argv),patch('run_commonsense_models.remove_weights') as remove:
                main();remove.assert_not_called()
            self.assertFalse(completed(root,'qwen',{r['audio_path'] for r in tasks}))
            (root/'results-qwen-commonsenseqa.jsonl').write_text(''.join(json.dumps(dict(audio_path=r['audio_path']))+'\n' for r in tasks))
            self.assertTrue(completed(root,'qwen',{r['audio_path'] for r in tasks}))


if __name__=='__main__':unittest.main()
