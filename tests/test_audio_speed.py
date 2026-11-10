"""Verify shared duration eligibility, conservative answer reuse, and speed-run resume."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compare_cloud_accuracy import PROMPT
from evaluate_audio_speed import main, reusable_answers, select_recordings


class SpeedTests(unittest.TestCase):
    def test_same_selected_questions_fit_every_rate_and_insufficient_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            records=[]
            durations={}
            for i,seconds in enumerate([14,16,15,8,23]):
                path=root/f'{i}.wav'
                path.write_bytes(b'audio')
                durations[str(path)]=seconds
                records.append(dict(clip_id=str(i),style='original',audio_path=path.name))
            records.insert(1,records[0])
            sf=SimpleNamespace(info=lambda p: SimpleNamespace(frames=durations[str(p)]*16000,samplerate=16000))
            rates=[i/10 for i in range(5,21)]
            selected=select_recordings(records,root,2,rates,sf)
            self.assertEqual([r['clip_id'] for r in selected],['0','2'])
            self.assertTrue(all(durations[r['audio_path']]/rate<=30 for r in selected for rate in rates))
            with self.assertRaisesRegex(ValueError,'found 3'):
                select_recordings(records,root,4,rates,sf)

    def test_reuse_requires_same_audio_recipe_seed_and_rate(self):
        settings=dict(seed=42,prompt=PROMPT,steps=128,block_length=32,max_new_tokens=512,
                      temperature=0.,decoder='fast_dllm',sample_rate=16000,
                      transform='librosa.effects.time_stretch; rate1 bypassed',librosa='0.11.0',max_seconds=30)
        record=dict(clip_id='c',question_id='q',audio_path='/same.wav',audio_bytes=20,audio_mtime_ns=1,answer='18')
        plan=dict(recordings=[record],rates=[.5,1.,2.],settings=settings)
        good=dict(clip_id='c',condition='baseline',rate=1.,seed=42,prompt=PROMPT,
                  reference_answer='18',response='Final answer: 18',correct=True)
        rows=[good,dict(good,rate=.7,condition='speed-0.7x'),dict(good,rate=2.,condition='speed-2x',seed=999)]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory)
            config=root/'run_config.json'
            config.write_text(json.dumps(plan))
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual([r['condition'] for r in reusable_answers(root,plan)],['baseline'])
            changed=dict(plan,recordings=[dict(record,audio_mtime_ns=2)])
            self.assertEqual(reusable_answers(root,changed),[])
            changed=dict(plan,settings=dict(settings,librosa='different'))
            self.assertEqual(reusable_answers(root,changed),[])
            changed=dict(plan,settings=dict(settings,
                transform='audiostretchy.core.stretch_audio; ratio=1/rate; rate1 bypassed',
                audiostretchy_version='1.3.14',gap_ratio='1/rate'))
            self.assertEqual(reusable_answers(root,changed),[])
            config.write_text(json.dumps(changed))
            self.assertEqual([r['condition'] for r in reusable_answers(root,changed)],['baseline'])
            different_version=dict(changed,settings=dict(changed['settings'],audiostretchy_version='other'))
            self.assertEqual(reusable_answers(root,different_version),[])

    def test_generation_bypasses_transform_at_1x_resets_seeds_and_resumes(self):
        class Audio:
            ndim=1
            def __init__(self,length): self.length=length
            def __len__(self): return self.length

        original=Audio(14*16000)
        transformed,generated,seeds,loads=[],[],[],[]
        def stretch(audio,rate):
            self.assertIs(audio,original)
            transformed.append(rate)
            return Audio(round(len(audio)/rate))
        def gen(audio,*args):
            generated.append(audio)
            return 'Final answer: 18'
        def load():
            loads.append(1)
            return None,None,None
        modules={'librosa':SimpleNamespace(__version__='0.11.0',effects=SimpleNamespace(time_stretch=stretch)),
                 'soundfile':SimpleNamespace(info=lambda p:SimpleNamespace(frames=len(original),samplerate=16000),
                                             read=lambda *a,**k:(original,16000)),
                 'numpy':SimpleNamespace(random=SimpleNamespace(seed=seeds.append)),
                 'torch':SimpleNamespace(manual_seed=seeds.append,cuda=SimpleNamespace(manual_seed_all=seeds.append)),
                 'extract_vectors':SimpleNamespace(load_model=load,gen=gen)}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory)
            (root/'clip.wav').write_bytes(b'audio')
            record=dict(clip_id='c',question_id='q',style='original',audio_path='clip.wav',answer='18')
            (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
            output=root/'result'
            args=['test','--path',str(root),'--num-examples','1','--rates','0.5','2', '--output',str(output)]
            with patch.dict('sys.modules',modules),patch.object(sys,'argv',args):
                main()
                main()
            self.assertEqual(transformed,[.5,2.])
            self.assertIs(generated[0],original)
            self.assertEqual([len(a)/16000 for a in generated],[14,28,7])
            self.assertEqual(seeds,[42]*9)
            self.assertEqual(loads,[1])
            config=json.loads((output/'run_config.json').read_text())
            self.assertEqual(config['rates'],[.5,1.,2.])
            self.assertIn('<=15s',config['settings']['selection'])
            with patch.object(sys,'argv',['test','--output',str(output),'--summary-only']):
                main()


if __name__=='__main__':
    unittest.main()
