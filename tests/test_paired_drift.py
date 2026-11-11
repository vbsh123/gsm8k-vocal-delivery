"""Verify saved-mean drift/outcome statistics and CPU-only handling."""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_paired_drift import analyze, number, outcome, vector_metrics


class OutcomeTests(unittest.TestCase):
    def test_outcomes_mean_natural_delivery_changes_not_steering(self):
        self.assertEqual(outcome(False,True),'recovery')
        self.assertEqual(outcome(True,False),'regression')
        self.assertEqual(outcome(True,True),'both_correct')
        self.assertEqual(outcome(False,False),'both_wrong')


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class SavedDriftTests(unittest.TestCase):
    def collection(self,root,missing=None):
        import torch
        folder=root/'collection'; (folder/'pairs').mkdir(parents=True)
        pairs=[]; manifest={}
        deltas=[[0.,1.],[0.,-1.],[.5,0.],[0.,0.]]
        responses=[('0','18'),('18','0'),('18','18'),('0','0')]
        for i,delta in enumerate(deltas):
            pair=dict(question_id=f'q{i}',speaker='Jenny',neutral_id=f'n{i}',angry_id=f'a{i}')
            pairs.append(pair)
            manifest[f'n{i}']=dict(answer='18');manifest[f'a{i}']=dict(answer='18')
            neutral=torch.tensor([1.,0.]).repeat(2,4,1)
            angry=neutral+torch.tensor(delta)
            if i!=missing:
                torch.save(dict(pair=pair,neutral=neutral,angry=angry,difference=angry-neutral,
                    neutral_response='Final answer: '+responses[i][0],angry_response='Final answer: '+responses[i][1],
                    neutral_forwards=20,angry_forwards=40,matched_forwards=20,averaging='matched'),folder/'pairs'/f'{i:05d}.pt')
        (folder/'collection-config.json').write_text(json.dumps(dict(pairs=pairs,manifest=manifest,averaging='matched',seed=42,
            groups=['acoustic','semantic','combined','answer'])))
        return folder

    def test_exact_norms_cosines_coherence_and_outcome_counts_without_cuda(self):
        import torch
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); collection=self.collection(root)
            with patch('torch.cuda._lazy_init',side_effect=AssertionError('No CUDA allowed')):
                metadata,summaries,comparisons=analyze(collection,root/'out')
            self.assertEqual(metadata['completed_pairs'],4)
            self.assertEqual(metadata['outcome_counts'],dict(recovery=1,regression=1,both_correct=1,both_wrong=1))
            self.assertEqual(metadata['neutral_correct'],2)
            self.assertEqual(metadata['angry_correct'],2)
            lookup={(r['layer'],r['group'],r['outcome']):r for r in summaries}
            recovery=lookup[0,'acoustic','recovery']
            self.assertAlmostEqual(recovery['mean_relative_l2_pct'],100)
            self.assertAlmostEqual(recovery['mean_cosine_similarity'],1/math.sqrt(2))
            self.assertAlmostEqual(recovery['direction_coherence'],1)
            self.assertAlmostEqual(lookup[0,'acoustic','all']['direction_coherence'],1/3)
            self.assertAlmostEqual(lookup[0,'acoustic','all']['mean_pairwise_shift_cosine'],-1/3)
            self.assertAlmostEqual(comparisons[0]['cosine_mean_recovery_vs_regression_shift'],-1)
            self.assertEqual(comparisons[0]['recovery_minus_regression_drift_pp'],0)
            with (root/'out/pair-metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),32)
            self.assertIn('not aligned',__import__('analyze_paired_drift').__doc__)
            self.assertIn('does not mean steering recovery',(root/'out/index.html').read_text())

    def test_partial_collection_is_explicitly_reported(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); collection=self.collection(root,missing=3)
            metadata,_,_=analyze(collection,root/'out')
            self.assertEqual(metadata['expected_pairs'],4)
            self.assertEqual(metadata['completed_pairs'],3)
            self.assertEqual(metadata['missing_pair_indices'],[3])

    def test_zero_length_directions_have_no_fabricated_angle_or_relative_change(self):
        import torch
        z=torch.zeros(1,1,2)
        _,units,valid,cosine,relative,norm=vector_metrics(z,z,torch)
        self.assertIsNone(number(cosine[0,0]))
        self.assertIsNone(number(relative[0,0]))
        self.assertFalse(valid.any())
        self.assertEqual(float(units.norm()),0)

    def test_difference_only_temporal_cache_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory); collection=self.collection(root)
            torch.save(dict(mean=torch.ones(3,2)),collection/'pairs/00000.pt')
            with self.assertRaisesRegex(ValueError,'differences only'):
                analyze(collection,root/'out')


if __name__=='__main__':
    unittest.main()
