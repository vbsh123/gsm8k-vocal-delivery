from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import run_phi4_gsm_L7 as run
import stop_phi4_nightly_opposite as stop
from steer import read_rows


class L7FollowupTests(unittest.TestCase):
    def test_full869_disjoint_vector_element_baselines_diagnostic_reuse_and_resume(self):
        for layer in (7,2):
            with self.subTest(layer=layer):self.check_layer(layer)

    def check_layer(self,layer):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'source';source.mkdir();out=root/'out'
            recipe=dict(model='Phi',revision='fixed',dtype='bfloat16')
            bank={f'q{i}':dict(question_id=f'q{i}',question='Q',answer='2') for i in range(869)}
            collection=[dict(question_id=f'train{i}') for i in range(400)]
            records=[dict(question_id=q,voice='Jenny',style=s,reference_answer='2',audio_file='audio',audio_sha256=q+s) for q in bank for s in ('neutral','angry')]
            original=dict(recordings=records,collection=collection,bank=bank,recipe=recipe,prompt='Same prompt')
            (source/'run_config.json').write_text(json.dumps(original))
            vector=torch.zeros(run.diagnostic.SHAPE);vector[layer,0]=3.
            torch.save(dict(angry_minus_neutral=vector,recipe=recipe,groups=['answer','audio'],
                run_config_sha256=run.digest(source/'run_config.json'),collection_question_ids=[p['question_id'] for p in collection]),source/'style-vectors.pt')
            refrows={}
            for r in records:
                refrows[r['question_id'],r['voice'],r['style']]=dict(r,response='#### 2',response_tokens=4,prompt='Same prompt',seed=42,max_new_tokens=512,model='Phi',revision='fixed')
            (source/'challenge.json').write_text('{}')
            prior=dict(refrows['q0','Jenny','neutral'],layer=layer,group='answer',strength=-1.,condition=f'neutral-L{layer}-answer-a-1',vector_sha256=run.digest(source/'style-vectors.pt'))
            before=(source/'run_config.json').read_bytes()
            with patch.object(run.diagnostic,'references',return_value=(refrows,refrows)), \
                 patch.object(run.diagnostic,'completed',return_value=([prior],{})), \
                 patch.object(run.opposite,'DEFAULT_OUTPUT',root/'no-opposite'):
                plan=run.prepare(source,out,layer);run.prepare(source,out,layer)
            self.assertEqual(len(plan['test']),869)
            self.assertFalse({p['question_id'] for p in plan['collection']} & {p['question_id'] for p in plan['test']})
            self.assertEqual(list(plan['strengths'].values()),[0.,-1.,-1.5])
            saved=read_rows(out/'results.jsonl');self.assertEqual(len(saved),870)
            self.assertEqual(sum(r['condition']=='neutral' for r in saved),869)
            self.assertEqual(sum(r['strength']==-1 for r in saved),1)
            torch.testing.assert_close(run.fixed_vector(plan,out),torch.full((3072,),3.))
            self.assertEqual((source/'run_config.json').read_bytes(),before)
            (source/'run_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):run.fixed_vector(plan,out)

    def test_scoped_stop_explicitly_cancels_only_l17_minus1point5_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'repo';repo.mkdir();proc=root/'proc';proc.mkdir()
            records=[(201,['bash','run_phi4_nightly_followups.sh']),
                (202,['bash','run_phi4_prepared_tasks.sh','--layer','17','--strength','-1.5']),
                (203,['python','phi4_task_steering.py','--mode','run','--layer','17','--strength','-1.5']),
                (204,['python','phi4_task_steering.py','--mode','run','--layer','7','--strength','-1.5']),
                (205,['python','phi4_task_steering.py','--mode','run','--layer','17','--strength','-1']),
                (206,['python','phi4_task_steering.py','--mode','summary','--layer','17','--strength','-1.5']),
                (208,['bash','run_phi4_gsm_L7.sh']),
                (209,['python','run_phi4_gsm_L7.py','--mode','run']),
                (210,['python','run_phi4_gsm_L7.py','--mode','worker','--worker-id','1']),
                (211,['python','run_phi4_gsm_L7.py','--mode','summary'])]
            for pid,args in records:
                path=proc/str(pid);path.mkdir();(path/'cwd').symlink_to(repo);(path/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args)+b'\0')
            self.assertEqual(stop.targets(repo,proc),[(201,True)])
            self.assertEqual(stop.targets(repo,proc,include_l17_tasks=True),[(201,True),(202,True),(203,False)])
            self.assertEqual(stop.targets(repo,proc,l7_only=True),[(208,True),(209,False),(210,False)])


if __name__=='__main__':unittest.main()
