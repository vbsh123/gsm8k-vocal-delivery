from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_active_pooling_probe as portable
import tests.test_diffa_active_pooling as active_tests


class PortablePoolingTests(unittest.TestCase):
    def fixture(self,root):
        source,bundle=active_tests.ActivePoolingTests().fixture(root)
        path=source/'run_config.json'
        plan=json.loads(path.read_text())
        split=dict(emotion2vec_predictions={q:dict(angry='angry') for q in plan['collection_ids']},
            audio_sha256={q:dict(neutral='neutral-'+q,angry='angry-'+q) for q in plan['collection_ids']})
        for pair in plan['pairs']:
            q=pair['question_id']
            for key,style in [('neutral','neutral'),('emotion','angry')]:
                pair[key].update(question_id=q,voice='Jenny',audio_file=str(root/(q+'-'+style+'.wav')),
                    audio_sha256=style+'-'+q)
        plan['split_sha256']=portable.vanilla.fingerprint(split)
        path.write_text(json.dumps(plan))
        (source.parent.parent/'split.json').write_text(json.dumps(split))
        state_path=source/'collection-stream.pt'
        state=torch.load(state_path,weights_only=True)
        state['run_config_sha256']=portable.winners.digest(path)
        for row in state['rows']:
            row.update(neutral_forwards=1,emotion_forwards=1,matched_forwards=1)
        torch.save(state,state_path)
        return source,bundle

    def test_portable_selection_is_agreed_disjoint_reproducible_and_self_contained(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            source,bundle=self.fixture(root)
            destination=root/'inputs.zip'
            portable.package(source,bundle,destination,questions=1)
            snapshot=destination.read_bytes()
            portable.package(source,bundle,destination,questions=1)
            self.assertEqual(destination.read_bytes(),snapshot)
            data=portable.load_bundle(destination)
            self.assertEqual(len(data['fit']['pairs']),1)
            self.assertEqual(data['fit']['collection_ids'],['fit0'])
            self.assertEqual([r['question_id'] for r in data['job']['target']['records']],['q0','q1'])
            self.assertFalse(set(data['fit']['collection_ids']) & {r['question_id'] for r in data['baseline']})
            self.assertEqual(data['agreed_labels']['fit0']['emotion'],'angry')
            # Portable preparation needs neither source files nor source checkpoint/vector paths.
            output=root/'Phi'
            plans,job=portable.prepare(data,destination,output)
            self.assertEqual(job['source']['vector_info']['pairs'],1)
            self.assertEqual(job['source']['extraction_ids'],['fit0'])
            self.assertEqual(job['target']['records'],data['job']['target']['records'])
            self.assertEqual(len(plans['active-block']['collection_ids']),1)
            receipt=output/'experiment.json'
            frozen=json.loads(receipt.read_text())
            frozen['runner_sha256']='previous runner before steering-policy update'
            receipt.write_text(json.dumps(frozen))
            before=portable.pooling.load_state(output,plans)['experiment_sha256']
            portable.prepare(data,destination,output)
            self.assertEqual(portable.pooling.load_state(output,plans)['experiment_sha256'],before)
            self.assertEqual(json.loads(receipt.read_text())['runner_sha256'],frozen['runner_sha256'])
            self.assertEqual(json.loads((output/'steering-policy.json').read_text())['steer_scopes'],['active-block'])
            bad=json.loads(json.dumps(data))
            bad['agreed_labels']['fit0']['emotion']='sad'
            with self.assertRaisesRegex(ValueError,'classification'):
                portable.prepare(bad,destination,root/'bad')
            bad=json.loads(json.dumps(data))
            bad['job']['target']['records'][0]['question_id']='fit0'
            with self.assertRaisesRegex(ValueError,'cohorts'):
                portable.prepare(bad,destination,root/'overlap')

    def test_subset_control_is_fitted_fresh_with_subset_count_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            source,bundle=self.fixture(root)
            destination=root/'inputs.zip'
            portable.package(source,bundle,destination,questions=1)
            data=portable.load_bundle(destination)
            output=root/'Phi'
            plans,job=portable.prepare(data,destination,output)
            old=torch.load(source/'style-vectors.pt',weights_only=True)
            delta=old['pair_mean_sums']['angry']/2
            paths={record['audio_sha256']:'/new/zip/'+record['audio_sha256']
                for pair in data['fit']['pairs'] for record in (pair['neutral'],pair['emotion'])}
            seen=[]
            def generate(record,*args):
                seen.append(record['audio_file'])
                return record['style'],{}
            def capture(model,generate,decoder,budget):
                style,_=generate()
                full=torch.ones(32,1,4,3)+(delta.unsqueeze(1) if style=='angry' else 0)
                current=torch.ones(32,1,3)*(9 if style=='angry' else 2)
                return '#### 2',{},full,current
            decoder=SimpleNamespace()
            modules={'src':SimpleNamespace(fast_dllm=SimpleNamespace(diffa_generate=decoder)),
                'src.fast_dllm':SimpleNamespace(diffa_generate=decoder),'src.fast_dllm.diffa_generate':decoder}
            with patch.dict('sys.modules',modules), \
                 patch.object(portable.vanilla.task,'generate',side_effect=generate), \
                 patch.object(portable.pooling,'capture_poolings',side_effect=capture) as capture_call:
                bad=json.loads(json.dumps(data))
                bad['source_rows']['fit0']['emotion_response']='different source answer'
                with self.assertRaisesRegex(ValueError,'replay differs'):
                    portable.collect(bad,plans,output,paths,None,None,None)
                self.assertFalse((output/'collection-checkpoint.pt').exists())
                portable.collect(data,plans,output,paths,None,None,None)
                portable.collect(data,plans,output,paths,None,None,None)
            self.assertEqual(capture_call.call_count,4)
            self.assertTrue(all(path.startswith('/new/zip/') for path in seen))
            full,info=portable.winners.read_vector(output/'full-buffer/style-vectors.pt','angry',28,'answer')
            current,_=portable.winners.read_vector(output/'active-block/style-vectors.pt','angry',28,'answer')
            self.assertEqual(info['pairs'],1)
            torch.testing.assert_close(full,torch.full((3,),2.))
            torch.testing.assert_close(current,torch.full((3,),7.))
            jobs=portable.active_steering_jobs(plans,job,data,output)
            self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0][1]['pooling_scope'],'active-block')
            control=portable.pooling.read_rows(output/'full-buffer/L31-answer-a1/results.jsonl')
            self.assertTrue(all(row['strength']==0. for row in control))
            for folder,selected in jobs:
                self.assertEqual(selected['source']['extraction_ids'],['fit0'])
                self.assertEqual(selected['source']['vector_info']['pairs'],1)
                self.assertEqual(selected['layer'],31)
                self.assertEqual(selected['strengths'],[0.,1.])
                self.assertEqual(portable.pooling.read_rows(folder/'results.jsonl'),data['baseline'])


if __name__=='__main__':
    unittest.main()
