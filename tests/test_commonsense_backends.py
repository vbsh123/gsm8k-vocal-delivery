from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class CommonsenseBackendTests(unittest.TestCase):
    def test_both_models_use_choice_reports_and_separate_result_files(self):
        from evaluate_commonsense_audio import PROMPT,score_choice
        with patch.dict('sys.modules',{'librosa':SimpleNamespace(),'soundfile':SimpleNamespace()}):
            import evaluate_qwen_omni as qwen
            import evaluate_kimi_audio as kimi
        for name,module in [('qwen',qwen),('kimi',kimi)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
                root=Path(d);tasks=[]
                for style in ('neutral','angry','happy','sad'):
                    audio=root/(style+'.wav')
                    with wave.open(str(audio),'wb') as f:
                        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*160)
                    tasks.append(dict(provider='azure',question_id='q',voice='Jenny',style=style,audio_path=audio.name,answer='B'))
                (root/'plan.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
                if name=='qwen':
                    recipe=dict(model=qwen.MODEL,revision=qwen.REVISION,transformers='4.52.4',prompt=PROMPT,
                        system_prompt=qwen.SYSTEM_PROMPT,max_new_tokens=512,seed=42,do_sample=False,dtype='bfloat16',
                        attention='sdpa',sample_rate=16000,max_audio_seconds=30,return_audio=False)
                    self.assertEqual(qwen.conversation(PROMPT)[1]['content'][1]['text'],PROMPT)
                else:
                    recipe=dict(model=kimi.MODEL,revision=kimi.REVISION,source_revision=kimi.SOURCE_REVISION,
                        tokenizer_model=kimi.TOKENIZER_MODEL,tokenizer_revision=kimi.TOKENIZER_REVISION,
                        transformers='4.51.3',prompt=PROMPT,max_new_tokens=512,seed=42,text_temperature=0.,audio_temperature=0.,
                        text_top_k=5,audio_top_k=5,text_repetition_penalty=1.,audio_repetition_penalty=1.,output_type='text',load_detokenizer=False)
                    self.assertEqual(kimi.messages(root/'neutral.wav',PROMPT)[0]['content'],PROMPT)
                (root/f'{name}-commonsenseqa-settings.json').write_text(json.dumps(recipe))
                values=[dict(t,**recipe,response='Final answer: B',**score_choice('Final answer: B','B')) for t in tasks]
                (root/f'results-{name}-commonsenseqa.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
                old=root/module.RESULTS;old.write_text('old math results remain unchanged')
                argv=['test','--task','commonsenseqa','--paths',str(root),'--num-questions','0','--summary-only']
                with patch('sys.argv',argv):module.main()
                self.assertEqual(old.read_text(),'old math results remain unchanged')
                report=json.loads((root/f'summary-{name}-commonsenseqa.json').read_text())
                self.assertEqual(report[0]['pairs'],1)
                self.assertEqual(report[0]['ref_correct'],1)
                self.assertEqual(report[0]['test_correct'],1)


if __name__=='__main__':unittest.main()
