from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
import diffa_happy_nightly as run
import diffa_stream_collection as stream


class StreamCollectionTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'full';(folder/'pairs').mkdir(parents=True)
        pairs=[dict(question_id=f'q{i}',neutral=dict(question_id=f'q{i}',style='neutral',answer='2'),
            emotion=dict(question_id=f'q{i}',style='happy',answer='2')) for i in range(4)]
        plan=dict(protocol=run.PROTOCOL,emotion='happy',name='ifeval',pairs=pairs,collection_ids=[p['question_id'] for p in pairs],
            prompt='eval',collection_prompt='original collection',max_new_tokens=1024,averaging='matched')
        (folder/'run_config.json').write_text(json.dumps(plan));sha=run.winners.digest(folder/'run_config.json')
        saved=[];generator=torch.Generator().manual_seed(42)
        for i,pair in enumerate(pairs):
            n=torch.randn(32,3,4,3,generator=generator)+3
            h=torch.randn(32,2,4,3,generator=generator)+4
            nm,hm,delta,matched=run.average_pair(n,h,'matched')
            row=dict(protocol=run.PROTOCOL,run_config_sha256=sha,question_id=pair['question_id'],neutral=nm,emotion=hm,difference=delta,
                neutral_response=f'neutral {i}',emotion_response=f'happy {i}',neutral_forwards=3,emotion_forwards=2,
                matched_forwards=matched,neutral_encoding=dict(mode='native'),emotion_encoding=dict(mode='native'))
            torch.save(row,folder/'pairs'/f'{i:05d}.pt');saved.append(row)
        return folder,plan,sha,saved

    def test_stream_vectors_and_all_drift_fields_equal_full_cache_after_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);full,plan,sha,saved=self.fixture(root)
            run.emotion.fit(plan,full)
            small=root/'stream';small.mkdir()
            state=stream.load(plan,small,sha)
            for row in saved[:2]:stream.append(state,row)
            run.save_tensor(torch,state,small/'collection-stream.pt')
            state=stream.load(plan,small,sha)
            for row in saved[2:]:stream.append(state,row)
            stream.finalize(plan,small,state)
            old=torch.load(full/'style-vectors.pt',weights_only=True);new=torch.load(small/'style-vectors.pt',weights_only=True)
            self.assertTrue(torch.equal(old['pair_mean_sums']['happy'],new['pair_mean_sums']['happy']))
            self.assertTrue(torch.equal(old['pair_counts']['happy'],new['pair_counts']['happy']))
            self.assertEqual(old['extraction_question_ids'],new['extraction_question_ids'])
            self.assertEqual(old['max_new_tokens'],new['max_new_tokens'])
            self.assertEqual(list(csv.DictReader(io.StringIO((full/'layer-drift.csv').read_text()))),list(csv.DictReader(io.StringIO((small/'layer-drift.csv').read_text()))))
            responses=run.read_rows(small/'collection-results.jsonl')
            self.assertEqual([r['response'] for r in responses],[text for i in range(4) for text in (f'neutral {i}',f'happy {i}')])
            self.assertTrue(all(r['max_new_tokens']==1024 and r['prompt']=='original collection' for r in responses))
            self.assertFalse((small/'pairs').exists())
            # Large activation means are absent from the per-question checkpoint rows.
            self.assertTrue(all('neutral' not in row and 'emotion' not in row and 'difference' not in row for row in state['rows']))

    def test_existing_pairs_and_final_vector_are_preserved_without_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            folder,plan,sha,_=self.fixture(Path(d));run.emotion.fit(plan,folder)
            hashes={p.name:run.winners.digest(p) for p in (folder/'pairs').glob('*.pt')}
            vector_hash=run.winners.digest(folder/'style-vectors.pt')
            with patch.object(run,'generate',side_effect=AssertionError('must reuse')),redirect_stdout(io.StringIO()):
                run.collect(plan,folder,None,None,None);run.collect(plan,folder,None,None,None)
            self.assertEqual(hashes,{p.name:run.winners.digest(p) for p in (folder/'pairs').glob('*.pt')})
            self.assertEqual(vector_hash,run.winners.digest(folder/'style-vectors.pt'))
            state=stream.load(plan,folder,sha);self.assertEqual(state['completed_ids'],plan['collection_ids'])

    def test_incomplete_or_wrong_checkpoint_cannot_be_used_as_complete_vector(self):
        with tempfile.TemporaryDirectory() as d:
            folder,plan,sha,saved=self.fixture(Path(d));state=stream.load(plan,folder,sha)
            stream.append(state,saved[0]);run.save_tensor(torch,state,folder/'collection-stream.pt')
            with self.assertRaisesRegex(ValueError,'incomplete'):stream.finalize(plan,folder,state)
            with self.assertRaises(ValueError):stream.load(plan,folder,'wrong-config')
            state['completed_ids']=['q1'];run.save_tensor(torch,state,folder/'collection-stream.pt')
            with self.assertRaises(ValueError):stream.load(plan,folder,sha)

    def test_new_pairs_write_only_running_checkpoint_and_resume_without_recapture(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,plan,_,_=self.fixture(root);folder=root/'new';folder.mkdir()
            (folder/'run_config.json').write_text(json.dumps(plan))
            traces=[]
            for i in range(4):
                traces.extend([(f'neutral {i}',{},torch.full((32,3,4,3),float(i+1))),
                    (f'happy {i}',{},torch.full((32,2,4,3),float(i+2)))])
            modules={name:ModuleType(name) for name in ('src','src.fast_dllm','src.fast_dllm.diffa_generate')}
            with patch.dict(sys.modules,modules),patch.object(run.emotion,'capture',side_effect=traces) as capture,redirect_stdout(io.StringIO()):
                run.collect(plan,folder,None,None,None);run.collect(plan,folder,None,None,None)
            self.assertEqual(capture.call_count,8)
            self.assertFalse((folder/'pairs').exists())
            state=torch.load(folder/'style-vectors.pt',weights_only=True)
            torch.testing.assert_close(state['pair_mean_sums']['happy'],torch.full((32,4,3),4.))
            self.assertEqual(len(run.read_rows(folder/'collection-results.jsonl')),8)

if __name__=='__main__':unittest.main()
