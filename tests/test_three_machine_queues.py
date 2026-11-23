import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import diffa_happy_nightly as happy
import phi4_zip_recheck as phi
from diffa_shared_three_queue import prepare_task, vector_ready
from report_three_machine_queues import collect, merge
from pack_three_machine_handoff import select_handoff, worker_matches


class ThreeMachineQueues(unittest.TestCase):
    def test_phi_uses_exact_exported_diffa_ids_and_settings(self):
        from export_diffa_cohorts import PROTOCOL as COHORT_PROTOCOL, make_cohort
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['diffa-train-b','diffa-train-a','diffa-test']
            def neutral(q):
                return dict(question_id=q,audio_sha256='zip-'+q+'neutral')
            plan = dict(name='svamp',seed=42,collection_ids=ids[:2],test_ids=ids[2:],
                        pairs=[dict(question_id=q,neutral=neutral(q)) for q in ids[:2]],
                        target=dict(records=[neutral(ids[2])],prompt='eval ####',max_new_tokens=512),
                        collection_prompt='collection Final answer',max_new_tokens=1024)
            bank = {q:dict(question='Question '+q,answer='2') for q in ids}
            cohort = dict(make_cohort(plan,bank),source_plan_sha256='receipt')
            data = root/'cohorts.json'
            data.write_text(json.dumps(dict(protocol=COHORT_PROTOCOL,datasets=dict(svamp=cohort))))
            def choose(name,q,style,**kwargs):
                self.assertEqual(kwargs,dict(question='Question '+q,answer='2',
                                            expected='zip-'+q+'neutral' if style=='neutral' else None))
                return dict(record=dict(question_id=q,voice='Jenny',audio_sha256='zip-'+q+style))
            registry = SimpleNamespace(choose=choose,options={})
            prepared=phi.prepare('svamp',data,registry,root)
            self.assertEqual(prepared['collection_ids'],ids[:2])
            self.assertEqual(prepared['test_ids'],ids[2:])
            self.assertEqual(prepared['prompt'],'eval ####')
            self.assertEqual(prepared['collection_prompt'],'collection Final answer')
            self.assertEqual(prepared['max_new_tokens'],512)
            self.assertEqual(prepared['collection_max_new_tokens'],1024)
            with self.assertRaises(ValueError):
                make_cohort(dict(plan,test_ids=ids[:1]),bank)
            data.write_text(json.dumps(dict(task='svamp',collection=[],test=[])))
            with self.assertRaisesRegex(ValueError,'exported original DIFFA'):
                phi.prepare('svamp',data,registry,root)

    def test_handoff_excludes_original_vectors_and_partial_aggregates(self):
        root=Path('/workspace/gsm8k-vocal-delivery')
        old=root/'outputs/original/style-vectors.pt'
        new=root/'outputs/diffa-gsm-shared-three-zeroshot-hashes/happy/style-vectors.pt'
        state=new.parent/'collection-stream.pt'
        config=new.parent/'run_config.json'
        chosen=select_handoff({old:b'old',new:b'new',state:b'partial',config:b'config'},root)
        self.assertEqual(set(chosen),{new,config})
        self.assertTrue(worker_matches(['python','-u','diffa_gsm_shared_emotions.py'],root,root))
        self.assertFalse(worker_matches(['python','diffa_happy_nightly.py'],root,root))
        self.assertFalse(worker_matches(['python','diffa_gsm_shared_emotions.py','--mode=summary'],root,root))
        self.assertFalse(worker_matches(['python','diffa_gsm_shared_emotions.py','--output=outputs/other'],root,root))

    def test_completed_vector_is_reused_without_collection_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=dict(emotion='happy',collection_ids=['train'],collection_prompt='#### prompt',max_new_tokens=512)
            torch.save(dict(pair_mean_sums={'happy':torch.ones(32,4,3)},pair_counts={'happy':torch.ones(32,dtype=torch.long)},
                            groups=happy.GROUPS,extraction_question_ids=['train'],prompt=plan['collection_prompt'],max_new_tokens=512),root/'style-vectors.pt')
            self.assertTrue(vector_ready(plan,root))
            self.assertFalse((root/'collection-stream.pt').exists())
            with self.assertRaises(ValueError):
                vector_ready(dict(plan,collection_ids=['other']),root)
    def test_shared_half_cap_same_ids_and_prompt_for_both_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            def choose(name, q, style, **kwargs):
                if kwargs.get('agreed') and int(q.rsplit('-',1)[1]) > 5:
                    return None
                record = dict(question_id=q, voice=happy.VOICE, style=style, reference_answer='2',
                              audio_file=f'/audio/{q}-{style}.wav', audio_sha256=q+style, seconds=1.)
                records[q+style] = dict(emotion=style)
                return dict(record=record, question='Question '+q, answer='2')
            ids = [f'svamp_chal-{i}' for i in range(10)]
            registry = SimpleNamespace(neutral_ids=lambda name: ids, choose=choose, labels=records)
            target = dict(bank={}, records=[], prompt='evaluation #### prompt')
            template = dict(name='svamp', target=target, bank={}, pairs=[], prompt='old eval', collection_prompt='old final')
            prepare_task('svamp', template, registry, root)
            a = json.loads((root/'angry/run_config.json').read_text())
            h = json.loads((root/'happy/run_config.json').read_text())
            self.assertEqual(a['collection_ids'], h['collection_ids'])
            self.assertEqual(a['test_ids'], h['test_ids'])
            self.assertEqual(len(a['collection_ids']), 5)
            self.assertEqual(len(a['test_ids']), 5)
            self.assertEqual(a['collection_prompt'], a['prompt'])
            self.assertEqual(a['prompt'], h['prompt'])

    def test_phi_fresh_deliveries_all_questions_before_steering_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def record(q, style):
                return dict(question_id=q, style=style, voice='Jenny', audio_sha256=q+style, reference_answer='2')
            pairs = [dict(question_id=q, neutral=record(q,'neutral'), angry=record(q,'angry')) for q in ('train','test')]
            plan = dict(task='svamp', prompt='eval prompt', collection_prompt='collection prompt',
                        collection_max_new_tokens=512, seed=42, recipe=dict(model='phi',revision='pinned'),
                        collection=pairs[:1], test=pairs[1:], max_new_tokens=1024,
                        deliveries=[dict(p[style], condition=style, split='collection' if p['question_id']=='train' else 'test')
                                    for p in pairs for style in ('neutral','angry')])
            (root/'run_config.json').write_text(json.dumps(plan))
            events = []
            def generate(native, row, prompt, **kwargs):
                events.append('steer' if kwargs.get('target') is not None else 'delivery')
                trace = dict(response='#### 2', response_tokens=3)
                if kwargs.get('collect'):
                    self.assertEqual(kwargs['capture_layers'],[0,11,30])
                    means = torch.zeros(3,2,3072)
                    if row['style']=='angry':
                        means[0,1],means[1,0],means[2,0] = 2.,3.,5.
                        means[0,0] = 99.  # Catch accidentally collecting L0 answer instead of audio.
                    trace.update(means=means, hook_forwards=3)
                if kwargs.get('target') is not None:
                    expected = {(0,'audio'):2.,(11,'answer'):3.,(30,'answer'):5.}[kwargs['target'],kwargs['group']]
                    torch.testing.assert_close(kwargs['direction'],torch.full((3072,),expected))
                    self.assertEqual(kwargs['strength'],-1.)
                self.assertEqual(kwargs['max_new_tokens'],512 if kwargs.get('collect') else 1024)
                self.assertEqual(prompt,'collection prompt' if kwargs.get('collect') else 'eval prompt')
                return trace
            def score(*args, **kwargs):
                events.append('report')
                (root/'summary.json').write_text(json.dumps(dict(verification=dict(differences=[]))))
                return SimpleNamespace(returncode=0)
            with patch('phi4_all_layer_diagnostic.generate', side_effect=generate), patch.object(phi.subprocess,'run',side_effect=score):
                phi.run_task(plan,root,None,phase='check')
                self.assertEqual(events,['delivery','delivery','delivery','delivery','report'])
                events.clear()
                phi.run_task(plan,root,None)
                self.assertEqual(events, ['report','steer','report'])
                events.clear()
                phi.run_task(plan,root,None)
                self.assertEqual(events,['report','report'])
                for target in ('L0-audio','L30-answer'):
                    phi.steer_task(plan,root,None,target)
                    phi.steer_task(plan,root,None,target)
                    self.assertEqual(len(phi.saved_rows(root/f'steering-{target}.jsonl',plan)),1)
            rows = [json.loads(s) for s in (root/'deliveries.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual({r['question_id'] for r in rows},{'train','test'})
            state = torch.load(root/'style-vectors.pt',map_location='cpu',weights_only=True)
            for key,value in zip(phi.TARGETS,(2.,3.,5.)):
                torch.testing.assert_close(state['directions'][key],torch.full((3072,),value))

    def test_three_workers_start_before_wait_and_cover_all_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory),datasets=['gsm8k','svamp'])
            events=[]
            def start(command,**kwargs):
                target=command[command.index('--worker-target')+1]
                self.assertEqual(command[command.index('--datasets')+1:],args.datasets)
                events.append(('start',target))
                def wait():
                    events.append(('wait',target))
                    return 0
                return SimpleNamespace(wait=wait)
            with patch.object(phi.subprocess,'Popen',side_effect=start):
                phi.launch_workers(args)
            self.assertEqual(events, [('start',k) for k in phi.TARGETS]+[('wait',k) for k in phi.TARGETS])

    def test_phi_report_keeps_direction_names_separate_from_scoring_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=dict(task='gsm8k',bank={'test':dict(question='Question',answer='2')},
                      collection=[],test=[dict(question_id='test')],deliveries=[{},{}],added_delivery_hashes=[])
            def saved(path,unused):
                conditions=('neutral','angry') if path.name=='deliveries.jsonl' else ('steered',)
                return {(c,'test'):dict(question_id='test',condition=c,response='#### 2',response_tokens=3) for c in conditions}
            def score(job,rows,folder):
                self.assertEqual(job['target']['bank'],plan['bank'])
                return [dict(r,metrics={'flexible-numeric':True}) for r in rows],{}
            with patch.object(phi,'saved_rows',side_effect=saved),patch.object(happy.winners,'score_rows',side_effect=score):
                for target in (None,*phi.TARGETS):
                    phi.report(plan,root,target)
                    filename=f'summary-{target}.json' if target else 'delivery-summary.json'
                    summary=json.loads((root/filename).read_text())
                    self.assertEqual(summary['steering_target'],target)
                    self.assertTrue((root/(f'rescored-{target}.jsonl' if target else 'delivery-rescored.jsonl')).exists())

    def test_report_uses_numeric_primary_not_flexible_for_math(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'diffa-happy-winners/gsm8k/strength-1/summary.json'
            path.parent.mkdir(parents=True)
            def result(metric,gain):
                return dict(strength=1.,metric=metric,gain_pp=gain,mcnemar_exact_p=.1,ci95_gain_pp=[0,5],pairs=10)
            path.write_text(json.dumps(dict(paired=[result('flexible-numeric',3.76),result('flexible-extract',3.2),result('strict-match',3.48)])))
            row = next(r for r in collect(root) if r['workflow']=='DIFFA original happy' and r['dataset']=='gsm8k')
            self.assertEqual(row['primary_gain_pp'],3.76)
            self.assertEqual(row['flexible_gain_pp'],3.2)
            combined=merge([dict(row,pairs=5),row,dict(workflow=row['workflow'],dataset='gsm8k',status='No saved summary')])
            self.assertEqual(combined,[row])


if __name__ == '__main__':
    unittest.main()
