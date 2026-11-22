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
            plan = dict(task='svamp', prompt='prompt', seed=42, recipe=dict(model='phi',revision='pinned'),
                        collection=pairs[:1], test=pairs[1:], max_new_tokens=1024,
                        deliveries=[dict(p[style], condition=style, split='collection' if p['question_id']=='train' else 'test')
                                    for p in pairs for style in ('neutral','angry')])
            (root/'run_config.json').write_text(json.dumps(plan))
            events = []
            def generate(native, row, prompt, **kwargs):
                events.append('steer' if kwargs.get('target') is not None else 'delivery')
                trace = dict(response='#### 2', response_tokens=3)
                if kwargs.get('collect'):
                    means = torch.zeros(1,2,3072)
                    means[0,0] = 1. if row['style']=='angry' else 0.
                    trace.update(means=means, hook_forwards=3)
                self.assertEqual(kwargs['max_new_tokens'],1024)
                return trace
            def score(*args, **kwargs):
                events.append('report')
                (root/'summary.json').write_text(json.dumps(dict(verification=dict(differences=[]))))
                return SimpleNamespace(returncode=0)
            with patch('phi4_all_layer_diagnostic.generate', side_effect=generate), patch.object(phi.subprocess,'run',side_effect=score):
                phi.run_task(plan,root,None,False,phase='check')
                self.assertEqual(events,['delivery','delivery','delivery','delivery','report'])
                events.clear()
                phi.run_task(plan,root,None,False)
                self.assertEqual(events, ['report','steer','report'])
                events.clear()
                phi.run_task(plan,root,None,False)
                self.assertEqual(events,['report','report'])
            rows = [json.loads(s) for s in (root/'deliveries.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual({r['question_id'] for r in rows},{'train','test'})
            vector = torch.load(root/'style-vectors.pt',map_location='cpu',weights_only=True)['angry_minus_neutral']
            torch.testing.assert_close(vector,torch.ones(3072))

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
