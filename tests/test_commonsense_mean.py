from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_commonsense_audio import PROMPT
from steer_commonsense_mean import split_recordings,main
from run_L31_timing import conditions


class CommonsenseMeanTests(unittest.TestCase):
    def records(self):
        return [dict(question_id=f'q{i:03}',voice='Jenny',style=s,answer='B',audio_path=f'{i}-{s}.wav')
                for i in range(500) for s in ('neutral','angry','happy','sad')]

    def test_fixed_split_is_disjoint_complete_and_independent_of_correctness(self):
        records=self.records();train,test=split_recordings(records,250)
        self.assertEqual((len(train),len(test)),(250,250))
        self.assertFalse({r['question_id'] for r in train}&{r['question_id'] for r in test})
        self.assertEqual(len({r['question_id'] for r in train}|{r['question_id'] for r in test}),500)
        self.assertTrue(all(r['style']=='neutral' for r in test))
        self.assertEqual(split_recordings(records,250),(train,test))

    def test_prepare_freezes_split_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);bench=root/'benchmark';bench.mkdir()
            (bench/'run_config.json').write_text(json.dumps(dict(prompt=PROMPT,seed=42,max_new_tokens=512,recordings=self.records())))
            with patch('sys.argv',['test','--benchmark',str(bench),'--output',str(root/'out'),'--prepare-only']):main()
            split=json.loads((root/'out/split.json').read_text())
            self.assertEqual((len(split['extraction']),len(split['test'])),(250,250))
            self.assertEqual(split['targets'],[dict(layer=12,group='acoustic',strength=1),dict(layer=31,group='answer',strength=1)])

    def test_strength_sweep_leaves_existing_timing_defaults_unchanged(self):
        self.assertEqual([r['schedule'] for r in conditions()[1:]],['early','middle','late'])
        rows=conditions(True)
        self.assertEqual([r['interventions'][0]['strength'] for r in rows[1:]],[.75,1.25,1.5])
        self.assertTrue(all('schedule' not in r for r in rows))


if __name__=='__main__':unittest.main()
