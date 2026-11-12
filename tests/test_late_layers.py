import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from run_jenny_late_layers import conditions, forward20_conditions, main, combined_report


class LateLayerConditionsTest(unittest.TestCase):
    def test_complete_individual_sweep(self):
        rows = conditions()
        self.assertEqual(rows[0], dict(name='baseline', interventions=[]))
        self.assertEqual(len(rows), 25)
        self.assertEqual(len({r['name'] for r in rows}), 25)
        actual = set()
        for row in rows[1:]:
            self.assertEqual(len(row['interventions']), 1)
            target = row['interventions'][0]
            actual.add((target['layer'], target['group'], target['strength'], row['schedule']))
        expected = {(l, g, a, s) for l in (29, 30, 31)
                    for g in ('acoustic', 'answer') for a in (0.5, 1)
                    for s in ('always', 'periodic2')}
        self.assertEqual(actual, expected)

    def test_forward20_periodic_is_separate_l12_condition(self):
        rows = forward20_conditions()
        self.assertEqual(rows, [dict(name='baseline', interventions=[]),
                               dict(name='L12-acoustic-a1-periodic2', schedule='periodic2',
                                    interventions=[dict(layer=12, group='acoustic', strength=1)])])

    def test_default_prepares_both_cohorts_and_extension_only_prepares_third(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            vector=root/'vectors.pt'; vector.touch()
            for name,config in [('second',dict(settings=dict(vectors=str(vector)),recordings=[dict(question_id='b')])),
                                ('third',dict(recordings=[dict(question_id='c')])),
                                ('forward20',dict(settings=dict(vectors=str(vector))))]:
                folder=root/name;folder.mkdir()
                (folder/'run_config.json').write_text(json.dumps(config))
            argv=['test','--source-run',str(root/'second'),'--third-source-run',str(root/'third'),
                  '--forward20-source',str(root/'forward20'),'--output',str(root/'out'),'--prepare-only']
            for extra,expected in [([],['second','second','third','third']),(['--third-only'],['third','third'])]:
                with patch.object(sys,'argv',argv+extra),patch('run_jenny_late_layers.subprocess.run') as run:
                    main()
                commands=[call.args[0] for call in run.call_args_list]
                self.assertEqual([Path(c[c.index('--source-run')+1]).name for c in commands],expected)
                self.assertTrue(all(c[-1]=='--prepare-only' for c in commands))

    def test_combined_summary_pools_disjoint_cohorts(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            for prefix,qid,bc,sc in [(root,'b',False,True),(root/'third-cohort','c',True,False)]:
                folder=prefix/'original_mean';folder.mkdir(parents=True)
                config=dict(settings=dict(conditions=conditions()),recordings=[dict(question_id=qid,clip_id=qid)])
                (folder/'run_config.json').write_text(json.dumps(config))
                rows=[dict(clip_id=qid,condition='baseline',correct=bc),
                      dict(clip_id=qid,condition='L29-acoustic-a0.5',correct=sc)]
                (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=combined_report(root)['original_mean'][0]
            self.assertEqual((result['pairs'],result['wins'],result['losses']), (2,1,1))
            path=root/'third-cohort/original_mean/run_config.json'
            config=json.loads(path.read_text());config['recordings'][0]['question_id']='b'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                combined_report(root)


if __name__ == '__main__':
    unittest.main()
