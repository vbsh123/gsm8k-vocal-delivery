import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_gsm_combo as combo
import phi4_l11_l30_parallel as run


class L11L30Tests(unittest.TestCase):
    def test_distinct_raw_answer_vectors_reach_their_own_hooks(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);record=dict(question_id='q',voice='Jenny',audio_sha256='audio',reference_answer='2')
            changes=[dict(layer=11,group='answer',strength=-1.),dict(layer=30,group='answer',strength=-2.)]
            plan=dict(recipe=dict(model='phi',revision='r'),prompt='prompt',max_new_tokens=512,
                      test=[record],conditions={'neutral':[],'combined':changes},sources=[])
            (out/'run_config.json').write_text(json.dumps(plan));torch.save({},out/'style-vectors.pt')
            baseline=dict(record,condition='neutral',run_config_sha256=combo.digest(out/'run_config.json'),interventions=[],
                          prompt='prompt',seed=42,max_new_tokens=512,model='phi',revision='r',response='#### 2',response_tokens=2)
            (out/'results.jsonl').write_text(json.dumps(baseline)+'\n')
            vectors={(11,'answer'):torch.ones(3),(30,'answer'):torch.ones(3)*7}
            with patch.object(combo.hooks,'generate',return_value=dict(response='#### 2',response_tokens=2)) as generate:
                combo.run(plan,out,vectors,native=object())
            changes=generate.call_args.kwargs['interventions']
            self.assertEqual([(x['layer'],x['strength']) for x in changes],[(11,-1.),(30,-2.)])
            torch.testing.assert_close(changes[0]['direction'],torch.ones(3))
            torch.testing.assert_close(changes[1]['direction'],torch.ones(3)*7)
            self.assertEqual(generate.call_count,1)

    def test_prepare_ifeval_keeps_separate_directions_budget_and_two_conditions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plans=[]
            base=dict(task='ifeval',collection=[dict(question_id='c')],recipe=dict(model='phi',revision='r'),
                      prompt='prompt',max_new_tokens=1024,bank={'t':{}},data_config_sha256='data',
                      recordings=[dict(question_id='t',condition='neutral')])
            for i,(layer,group,_) in enumerate(run.COMBOS['L0-L11-L30']):
                folder=root/f'L{layer}';folder.mkdir();plan=dict(base,layer=layer,group=group)
                (folder/'run_config.json').write_text(json.dumps(plan))
                torch.save(dict(run_config_sha256=run.digest(folder/'run_config.json'),collection_question_ids=['c'],
                    layer=layer,group=group,model='phi',revision='r',angry_minus_neutral=torch.ones(3072)*(i+1)),folder/'style-vectors.pt')
                plans.append((plan,folder))
            plan,vectors=run.prepare_combo(root/'combined',plans,run.COMBOS['L0-L11-L30'])
            self.assertEqual(plan['max_new_tokens'],1024)
            self.assertEqual(set(plan['conditions']),{'neutral','combined'})
            self.assertEqual(set(vectors),{(0,'audio'),(11,'answer'),(30,'answer')})
            self.assertEqual([r['strength'] for r in plan['conditions']['combined']],[-1.,-1.,-2.])
            torch.testing.assert_close(vectors[11,'answer'],torch.ones(3072)*2)
            torch.testing.assert_close(vectors[30,'answer'],torch.ones(3072)*3)

    def test_parent_starts_two_disjoint_combo_workers_without_gpu_queue(self):
        with tempfile.TemporaryDirectory() as d:
            import sys
            with patch.object(sys,'argv',['runner','--datasets','gsm8k','bbh5','--output',d]), \
                 patch.object(run,'source_plans') as sources,patch.object(run,'run_workers') as launch, \
                 patch('phi4_queued_workers.gpu_queue',side_effect=AssertionError('No GPU wait allowed')):
                run.main()
            self.assertEqual(sources.call_count,2)
            commands=launch.call_args.args[0]
            self.assertEqual(len(commands),2)
            self.assertEqual([c[c.index('--worker-id')+1] for c in commands],['0','1'])
            self.assertTrue(all(c[c.index('--datasets')+1:]==['gsm8k','bbh5'] for c in commands))


if __name__=='__main__':unittest.main()
