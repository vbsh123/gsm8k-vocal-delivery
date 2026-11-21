from contextlib import contextmanager,redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import phi4_svamp_l11_combo as run
from kimi_bbh_layer_diagnostic import digest


class L11ComboTests(unittest.TestCase):
    def test_subset_pair_reuse_then_collects_only_missing_question(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'subset';(source/'pairs').mkdir(parents=True)
            pairs=[dict(question_id=f'q{i}',voice='Jenny',**{style:dict(question_id=f'q{i}',voice='Jenny',style=style,audio_sha256=style+str(i)) for style in ('neutral','angry')}) for i in range(3)]
            plan=dict(task='svamp',collection=pairs,recipe=dict(model='phi',revision='rev'),prompt='same',seed=42,max_new_tokens=512,
                layer=11,group='answer',vector_definition='same')
            previous=run.collection_config(dict(plan,collection=pairs[:2]));(source/'run_config.json').write_text(json.dumps(previous))
            for i,pair in enumerate(pairs[:2]):
                traces={style:dict(mean=torch.full((3072,),1. if style=='neutral' else 3.),hook_forwards=1,response='#### 2',response_tokens=3) for style in ('neutral','angry')}
                torch.save(dict(protocol=previous['protocol'],run_config_sha256=digest(source/'run_config.json'),question_id=pair['question_id'],voice='Jenny',
                    audio_sha256={s:pair[s]['audio_sha256'] for s in traces},traces=traces,answer_difference=torch.full((3072,),2.)),source/'pairs'/f'{i:04d}.pt')
            before=[digest(p) for p in sorted((source/'pairs').glob('*.pt'))]
            with redirect_stdout(io.StringIO()):count=run.reuse_subset_pairs(plan,root/'cache',source)
            self.assertEqual(count,2)
            cache=root/'cache/svamp/L11-answer';calls=[]
            def generate(row,prompt,observe):
                calls.append((row['question_id'],row['style']))
                return dict(mean=torch.full((3072,),1. if row['style']=='neutral' else 3.),hook_forwards=1,response='#### 2',response_tokens=3)
            config=json.loads((cache/'run_config.json').read_text())
            with redirect_stdout(io.StringIO()):vector=run.task.engine.collect(config,cache,SimpleNamespace(generate=generate),digest(cache/'run_config.json'))
            self.assertEqual(calls,[('q2','neutral'),('q2','angry')])
            torch.testing.assert_close(vector,torch.full((3072,),2.))
            self.assertEqual(before,[digest(p) for p in sorted((source/'pairs').glob('*.pt'))])
            self.assertEqual(torch.load(cache/'style-vectors.pt',weights_only=True)['collection_question_ids'],['q0','q1','q2'])
            changed=dict(plan,prompt='different')
            self.assertEqual(run.reuse_subset_pairs(changed,root/'other',source),0)

    def test_run_waits_then_collects_vectors_and_generates_combination_only(self):
        events=[]
        @contextmanager
        def queued():events.append('admitted');yield;events.append('released')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data_root=root/'data';(data_root/'svamp').mkdir(parents=True)
            (data_root/'svamp/data_config.json').write_text(json.dumps(dict(collection=[{}]*500,test=[{}]*500)))
            plans=[(dict(layer=layer,group=group),root/group) for layer,group in ((0,'audio'),(11,'answer'))]
            model=object();native=SimpleNamespace(native=model,plan=None)
            def load(plan):self.assertEqual(events,['admitted']);events.append('model');return native
            def collect(plan,*args):events.append(('vector',plan['layer'],plan['group']))
            def generate(plan,out,vectors,**kwargs):
                self.assertIs(kwargs['native'],model);events.append('combined')
            with patch.object(sys,'argv',['l11','--data-root',str(data_root),'--output',str(root/'out')]), \
                patch.object(run.collection,'sources',return_value=plans) as sources,patch.object(run,'gpu_queue',queued), \
                patch('evaluate_bbh_candidates.concurrent_budget_now'),patch.object(run.task,'Native',side_effect=load), \
                patch.object(run,'reuse_subset_pairs'),patch.object(run.task,'collection_vector',side_effect=collect), \
                patch.object(run.task,'evaluate',side_effect=AssertionError('no standalone evaluation')), \
                patch.object(run.combo,'prepare',return_value=({},{})) as prepare,patch.object(run.combo,'run',side_effect=generate):
                run.main()
            self.assertEqual(sources.call_args.kwargs['settings'],((0,'audio',-1.),(11,'answer',-1.)))
            self.assertEqual(prepare.call_args.kwargs['answer_layer'],11)
            self.assertEqual(prepare.call_args.kwargs['answer_strength'],-1.)
            self.assertTrue(prepare.call_args.kwargs['combined_only'])
            self.assertEqual(events,['admitted','model',('vector',0,'audio'),('vector',11,'answer'),'combined','released'])

if __name__=='__main__':unittest.main()
