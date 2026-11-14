from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from run_commonsense_models import remove_weights,completed,main


class CommonsenseQueueTests(unittest.TestCase):
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
