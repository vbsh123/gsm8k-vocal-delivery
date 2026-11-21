import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_gsm_combo as combo
import phi4_l0_l11_parallel as run


class L11ParallelTests(unittest.TestCase):
    def test_assignments_are_disjoint_and_cover_requested_tasks(self):
        self.assertEqual(run.ASSIGNMENT[0], ('multiarith','gsm8k'))
        self.assertEqual(run.ASSIGNMENT[1], ('ifeval','bbh3','bbh5'))
        self.assertFalse(set(run.ASSIGNMENT[0]) & set(run.ASSIGNMENT[1]))
        self.assertEqual(run.SETTINGS, ((0,'audio',-1.),(11,'answer',-1.)))

    def test_worker_reuses_one_model_and_generates_only_combined(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(worker_id=1,output=Path(d),collection_cache=Path(d)/'cache')
            def sources(name,args):
                return [(dict(task=name,layer=layer,group=group),args.output/name/group)
                        for layer,group,_ in run.SETTINGS]
            native=SimpleNamespace(native=object())
            with patch.object(run,'source_plans',side_effect=sources), \
                 patch('evaluate_bbh_candidates.concurrent_budget_now'), \
                 patch.object(run.task,'Native',return_value=native) as load, \
                 patch.object(run.task,'collection_vector') as collect, \
                 patch.object(run.combo,'prepare',return_value=({},{})) as prepare, \
                 patch.object(run.combo,'run') as generate:
                run.worker(args)
            self.assertEqual(load.call_count,1)
            self.assertEqual(collect.call_count,6)
            self.assertEqual(generate.call_count,3)
            self.assertEqual([c.kwargs['dataset'] for c in prepare.call_args_list], ['ifeval','bbh3','bbh5'])
            self.assertTrue(all(c.kwargs['combined_only'] for c in prepare.call_args_list))
            self.assertTrue(all(c.kwargs['native'] is native.native for c in generate.call_args_list))

    def test_ifeval_combination_preserves_1024_budget_and_official_reporting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folders=[];recipe=dict(model='phi',revision='revision')
            collection=[dict(question_id=f'c{i}') for i in range(254)]
            rows=[dict(question_id=f't{i}',voice='Jenny',condition='neutral',audio_sha256=str(i),reference_answer=None)
                  for i in range(254)]
            for layer,group in [(0,'audio'),(11,'answer')]:
                folder=root/group;folder.mkdir();folders.append(folder)
                config=dict(collection=collection,recipe=recipe,prompt='Follow instructions.',seed=42,
                            max_new_tokens=1024,recordings=rows,bank={r['question_id']:{} for r in rows})
                (folder/'run_config.json').write_text(json.dumps(config))
                torch.save(dict(run_config_sha256=combo.digest(folder/'run_config.json'),
                                collection_question_ids=[p['question_id'] for p in collection],layer=layer,group=group,
                                **recipe,angry_minus_neutral=torch.ones(3072)),folder/'style-vectors.pt')
            plan,vectors=combo.prepare(root/'combined',*folders,dataset='ifeval',combined_only=True,
                                       answer_layer=11,answer_strength=-1.,audio_strength=-1.)
            self.assertEqual(plan['max_new_tokens'],1024)
            self.assertEqual(set(plan['conditions']), {'neutral','combined'})
            self.assertEqual(plan['conditions']['combined'], [dict(layer=0,group='audio',strength=-1.),dict(layer=11,group='answer',strength=-1.)])
            with patch('phi4_ifeval.report') as official:
                combo.report(plan,root/'combined')
            self.assertEqual(official.call_count,1)
            self.assertEqual(set(r['condition'] for r in official.call_args.args[0]['recordings']), {'neutral','combined'})
            row=dict(rows[0],run_config_sha256='hash',prompt=plan['prompt'],seed=42,max_new_tokens=1024,
                     **recipe,interventions=[])
            self.assertTrue(combo.compatible(row,'hash',plan,rows[0],[]))
            self.assertFalse(combo.compatible(dict(row,max_new_tokens=512),'hash',plan,rows[0],[]))


if __name__=='__main__':unittest.main()
