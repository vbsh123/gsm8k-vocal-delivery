"""Verify frozen promotion choices, separate directions and periodic3 block timing."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from steer import schedule_active
from run_jenny_promoted_third import main


class PromotedThirdTests(unittest.TestCase):
    def test_periodic3_fires_on_first_fourth_seventh_and_restarts_each_block(self):
        active=[schedule_active('periodic3',dict(block_forward=i,forward=100+i)) for i in range(1,8)]
        self.assertEqual(active,[True,False,False,True,False,False,True])
        self.assertTrue(schedule_active('periodic3',dict(block_forward=1,forward=109)))
        self.assertEqual([schedule_active('periodic2',dict(block_forward=i)) for i in range(1,5)],
                         [True,False,True,False])

    def test_prepare_preserves_distinct_checkpoints_schedules_and_baseline_sources(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            for name,folder in [
                ('original','outputs/jenny-large/timing/original_mean'),
                ('angry','outputs/steer-jenny-outcome-filters/matched/angry_correct_mean'),
                ('windows','outputs/jenny-large/temporal-evaluation/forward_windows')]:
                vector=root/f'{name}.pt'; vector.touch()
                source=root/folder; source.mkdir(parents=True)
                (source/'run_config.json').write_text(json.dumps(dict(settings=dict(vectors=str(vector)),recordings=[dict(clip_id='old1'),dict(clip_id='old2')])))
            data=root/'audio'; data.mkdir()
            tasks=[dict(provider='azure',style='neutral',voice='Jenny',question_id=f'q{i}',audio_path=f'{i}.wav',answer='18') for i in range(3)]
            for row in tasks:
                (data/row['audio_path']).touch()
            (data/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
            before=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys,'argv',['test','--path',str(data),'--prepare-only']),patch.dict(sys.modules,{'soundfile':SimpleNamespace(info=lambda path:SimpleNamespace(duration=10))}),patch('run_jenny_promoted_third.subprocess.run') as execute:
                    main()
                commands=[c.args[0] for c in execute.call_args_list]
                self.assertEqual(len(commands),4)
                self.assertTrue(all(c[-1]=='--prepare-only' for c in commands))
                self.assertEqual([Path(c[c.index('--vectors')+1]).stem for c in commands],['original','angry','windows','original'])
                output=root/'outputs/steer-jenny-third-promoted'
                original=json.loads((output/'original-conditions.json').read_text())
                self.assertEqual([r.get('schedule','always') for r in original],['always','always','periodic2','periodic3'])
                self.assertEqual(commands[-1][commands[-1].index('--num-examples')+1],'2')
                self.assertEqual(len(json.loads((output/'cohort/run_config.json').read_text())['recordings']),3)
                self.assertEqual(commands[1][commands[1].index('--baseline-source')+1],'outputs/steer-jenny-third-promoted/original')
            finally:
                os.chdir(before)


if __name__=='__main__':
    unittest.main()
