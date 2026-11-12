from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_L31_forward20_after_current as queue


class L31PipelineTests(unittest.TestCase):
    def test_waits_for_parent_then_collects_and_evaluates_both_cohorts(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            (root/'outputs').mkdir()
            audio=root/'clip.wav';audio.touch()
            vector=root/'original.pt';vector.touch()
            original=root/'late/original_mean';original.mkdir(parents=True)
            (original/'run_config.json').write_text(json.dumps(dict(settings=dict(vectors=str(vector)),
                recordings=[dict(clip_id='b',question_id='b',audio_path=str(audio),answer='1')])) )
            collection=root/'collection';collection.mkdir()
            (collection/'collection-config.json').write_text(json.dumps(dict(
                pairs=[dict(question_id='train',neutral_id='n',angry_id='a')],
                manifest={'n':dict(audio_path=str(audio)),'a':dict(audio_path=str(audio))})))
            third=root/'third';third.mkdir()
            (third/'run_config.json').write_text(json.dumps(dict(recordings=[
                dict(clip_id='c',question_id='c',audio_path=str(audio),answer='1')])))
            argv=['test','--late-root',str(root/'late'),'--collection',str(collection),
                  '--third-source-run',str(third),'--output',str(root/'new')]
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys,'argv',argv),patch.object(queue.subprocess,'run') as run,\
                     patch.object(queue.subprocess,'check_output',return_value=''),\
                     patch.object(queue,'reuse_third_baselines'),patch.object(queue,'pooled_report',return_value={}):
                    queue.main()
            finally:
                os.chdir(previous)
            commands=[c.args[0] for c in run.call_args_list]
            self.assertEqual(commands[2][-1],'finalist-queue')
            self.assertEqual(commands[3][-1],'azure-emotions')
            self.assertEqual(commands[4][2],'collect_temporal_vectors.py')
            self.assertEqual(commands[4][commands[4].index('--targets')+1],'31:answer')
            self.assertEqual(commands[4][-1],'forward_windows')
            self.assertEqual(commands[5][-3:],['--conditions','L31-answer-a1','L31-answer-a1-periodic2'])
            self.assertEqual([Path(c[c.index('--source-run')+1]).name for c in commands[6:8]],['original_mean','third'])
            self.assertTrue(all('run_jenny_late_layers.py' not in c for c in commands))
            status=json.loads((root/'new/status.json').read_text())
            self.assertEqual(status['state'],'finished')


if __name__=='__main__':
    unittest.main()
