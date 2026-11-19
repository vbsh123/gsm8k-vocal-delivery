import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_multiarith_guy_emotions as run
from steer import read_rows


class GuyEmotionTests(unittest.TestCase):
    def baseline(self,root):
        folder=root/'baseline';folder.mkdir()
        bank={f'q{i}':dict(question_id=f'q{i}',question='Q',answer='2') for i in range(290)}
        records=[dict(question_id=q,voice='Guy',style=s,condition=s,audio_file='audio',audio_sha256=q+s,answer='2')
                 for q in bank for s in ('neutral','angry')]
        plan=dict(task='multiarith',voice='en-US-GuyNeural',decoder='fast_dllm',seed=42,max_new_tokens=512,
            steps=128,block_length=32,temperature=0,prompt='Exact original prompt',bank=bank,recordings=records)
        (folder/'run_config.json').write_text(json.dumps(plan))
        rows=[dict(r,response='#### 2',reference_answer='2',prompt=plan['prompt'],seed=42,max_new_tokens=512,decoder='fast_dllm',interventions=[]) for r in records]
        (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        audio={(q,s):dict(question_id=q,voice='Guy',style=s,audio_file='new-audio',audio_sha256=q+s,answer='2')
            for q in bank for s in ('happy','sad')}
        return folder,audio

    def test_only_new_emotions_generated_original_answers_unchanged_and_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);baseline,audio=self.baseline(root)
            before={p.name:p.read_bytes() for p in baseline.iterdir()}
            args=SimpleNamespace(baseline=baseline,source=root/'audio',output=root/'four')
            with patch.object(run.subprocess,'run') as tts,patch.object(run.reasoning,'source_records',return_value=audio):
                plan=run.prepare(args);run.prepare(args)
            command=tts.call_args.args[0]
            self.assertEqual(command[command.index('--styles')+1:command.index('--output')],['happy','sad'])
            self.assertEqual(command[command.index('--azure-voice')+1],'en-US-GuyNeural')
            self.assertEqual(len(plan['recordings']),1160)
            self.assertEqual(plan['prompt'],'Exact original prompt')
            rows=read_rows(args.output/'results.jsonl')
            self.assertEqual(len(rows),580)
            self.assertEqual({r['condition'] for r in rows},{'neutral','angry'})
            done={(r['question_id'],r['voice'],r['condition']) for r in rows}
            pending=[r for r in plan['recordings'] if (r['question_id'],r['voice'],r['condition']) not in done]
            self.assertEqual(len(pending),580)
            self.assertEqual({r['condition'] for r in pending},{'happy','sad'})
            self.assertEqual(before,{p.name:p.read_bytes() for p in baseline.iterdir()})

    def test_partial_original_baseline_rejected_before_tts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);baseline,_=self.baseline(root)
            (baseline/'results.jsonl').write_text('')
            args=SimpleNamespace(baseline=baseline,source=root/'audio',output=root/'four')
            with patch.object(run.subprocess,'run') as tts:
                with self.assertRaisesRegex(ValueError,'Finish all580'):run.prepare(args)
                tts.assert_not_called()


if __name__=='__main__':unittest.main()
