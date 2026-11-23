import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from evaluate_gsm_candidates import EXPLICIT,ORIGINAL
import diffa_gsm_prompt_ablation as ablation
from pack_diffa_gsm_prompt_inputs import inputs,MANIFEST
from restore_diffa_bundle import restore


class PromptAblationTests(unittest.TestCase):
    def vector(self,path,emotion,count,ids,value):
        torch.save(dict(mode='fixed',method='mean',groups=['answer'],prompt='collection prompt',max_new_tokens=512,
                        extraction_question_ids=ids,pair_counts={emotion:torch.full((32,),count)},
                        pair_mean_sums={emotion:torch.full((32,1,3),value*count)}),path)

    def fixture(self,repo):
        original=repo/'outputs/diffa-happy-winners/gsm8k';original.mkdir(parents=True)
        shared=repo/'outputs/diffa-gsm-shared-three-zeroshot-hashes';shared.mkdir()
        audio=repo/'neutral.wav';audio.write_bytes(b'neutral-audio')
        legacy=repo/'outputs/legacy-gsm';legacy.mkdir()
        source=legacy/'run_config.json';source.write_text(json.dumps(dict(source='legacy-path')))
        train=[f'train-{i}' for i in range(425)]
        ids=[f'q-{i}' for i in range(718)]
        records=[dict(question_id=q,voice='Jenny',style='neutral',audio_file=str(audio),audio_sha256=ablation.winners.digest(audio),reference_answer='2') for q in ids]
        angry=legacy/'style-vectors.pt';self.vector(angry,'angry',425,train,2.)
        target=dict(name='gsm8k',task='gsm8k',records=records,bank={q:dict(question='question',answer='2') for q in ids},
                    prompt=EXPLICIT,seed=42,max_new_tokens=512,steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                    vector=str(angry),vector_sha256=ablation.winners.digest(angry),config_file=str(source),run=str(legacy),reuse_runs=[],
                    extraction_ids=train,vector_info={})
        plan=dict(emotion='happy',name='gsm8k',collection_ids=train,test_ids=ids,target=target)
        (original/'run_config.json').write_text(json.dumps(plan))
        self.vector(original/'style-vectors.pt','happy',425,train,7.)
        for emotion,value in [('angry',5.),('happy',9.)]:
            folder=shared/emotion;folder.mkdir()
            shared_train=[f'shared-train-{i}' for i in range(609)]
            shared_target=dict(target,records=records[:660],bank={q:target['bank'][q] for q in ids[:660]})
            shared_plan=dict(plan,emotion=emotion,collection_ids=shared_train,test_ids=ids[:660],target=shared_target)
            (folder/'run_config.json').write_text(json.dumps(shared_plan))
            self.vector(folder/'style-vectors.pt',emotion,609,shared_train,value)
        return original,shared

    def test_both_protocols_keep_vectors_ids_audio_and_only_change_reasoning_sentence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,shared=self.fixture(root)
            for cohort,source,size,values in [('original',original,718,(2.,7.)),('shared3',shared,660,(5.,9.))]:
                plan=ablation.prepare(cohort,source,root/'out'/cohort)
                self.assertEqual(plan['prompts'],dict(concise=EXPLICIT,older=ORIGINAL))
                self.assertEqual(len(plan['test_ids']),size)
                self.assertEqual(len(ablation.pending(plan,{})),size*6)
                self.assertEqual(plan['target']['max_new_tokens'],512)
                self.assertTrue(all(r['style']=='neutral' for r in plan['target']['records']))
                for emotion,value in zip(('angry','happy'),values):
                    direction,_=ablation.winners.read_vector(plan['vectors'][emotion]['vector'],emotion)
                    torch.testing.assert_close(direction,torch.full((3,),value))
            packed=inputs(root)
            self.assertIn(original/'style-vectors.pt',packed)
            self.assertTrue(any(p.name=='style-vectors.pt' and p.parent.name=='legacy-gsm' for p in packed))
            self.assertFalse(any(p.suffix=='.wav' for p in packed))

    def test_reuse_is_bound_to_prompt_vector_hash_and_exact_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,shared=self.fixture(root)
            folder=root/'out/original'
            plan=ablation.prepare('original',original,folder)
            record=plan['target']['records'][0]
            def row(condition,strength,vector_hash=None):
                return dict(record,response='#### 2',response_tokens=3,prompt=EXPLICIT,condition=condition,
                            strength=strength,vector_sha256=vector_hash,
                            interventions=[dict(layer=31,group='answer',strength=1.)] if strength else [],
                            **{f:plan['target'][f] for f in ('seed','max_new_tokens','steps','block_length','temperature','decoder')})
            legacy=Path(plan['target']['run'])
            oldrows=[row('neutral',0.),row('L31-answer-a1',1.,plan['vectors']['angry']['vector_sha256']),
                     row('angry',0.)]
            (legacy/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in oldrows))
            happy=original/'strength-1';happy.mkdir()
            (happy/'run_config.json').write_text(json.dumps(dict(source=dict(vector_sha256=plan['vectors']['happy']['vector_sha256']))))
            (happy/'results.jsonl').write_text(json.dumps(row('L31-answer-a1',1.,plan['vectors']['happy']['vector_sha256']))+'\n')
            done=ablation.reuse(plan,folder)
            self.assertEqual({k[2:] for k in done},{('concise',c) for c in ablation.CONDITIONS})
            self.assertFalse(any(k[2]=='older' for k in done))
            self.assertEqual(len(ablation.answers(plan,folder)),3)
            old=ablation.copied(record,oldrows[0],plan,folder,'older','neutral','generated')
            with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(old)+'\n')
            shared_folder=root/'out/shared3'
            shared_plan=ablation.prepare('shared3',shared,shared_folder)
            shared_done=ablation.reuse(shared_plan,shared_folder,[folder])
            self.assertEqual({k[2:] for k in shared_done},{('concise','neutral'),('older','neutral')})
            bad=dict(old,prompt=EXPLICIT)
            with self.assertRaisesRegex(ValueError,'provenance'):
                ablation.validate(bad,plan,ablation.winners.digest(folder/'run_config.json'))

    def test_supplemental_restore_does_not_replace_main_migration_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repo';root.mkdir()
            main=root/'DIFFA_MIGRATION_MANIFEST.json';main.write_text('original receipt')
            archive=Path(directory)/'inputs.zip'
            with zipfile.ZipFile(archive,'w') as out:
                out.writestr(MANIFEST,json.dumps(dict(repo=str(root))))
                out.writestr('outputs/original-vector.txt','payload')
            restore(archive,root)
            self.assertEqual(main.read_text(),'original receipt')
            self.assertEqual((root/'outputs/original-vector.txt').read_text(),'payload')

    def test_reports_keep_prompt_emotion_and_common_cohort_comparisons_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,shared=self.fixture(root)
            output=root/'out'
            for cohort,source in [('original',original),('shared3',shared)]:
                folder=output/cohort
                plan=ablation.prepare(cohort,source,folder)
                rows=[]
                for record in plan['target']['records'][:2]:
                    for prompt in plan['prompts']:
                        for condition in ablation.CONDITIONS:
                            response='yes' if condition==('happy-vector' if cohort=='original' else 'angry-vector') else 'no'
                            rows.append(ablation.copied(record,dict(response=response,response_tokens=2),plan,folder,prompt,condition,'fixture'))
                (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            def score(job,rows,folder):
                return [dict(r,metrics={'numeric':r['response']=='yes'}) for r in rows],{}
            with patch.object(ablation.winners,'score_rows',side_effect=score):
                for cohort in ('original','shared3'):ablation.report(output/cohort)
                ablation.report_common(output)
            summary=json.loads((output/'original/summary.json').read_text())
            direct=[r for r in summary['paired'] if r['baseline']=='angry-vector' and r['test']=='happy-vector']
            self.assertEqual(len(direct),2)
            self.assertTrue(all(r['gain_pp']==100. for r in direct))
            self.assertEqual(len(summary['prompt_effects']),3)
            common=json.loads((output/'common-heldout-comparison.json').read_text())
            self.assertEqual(len(common['paired']),6)
            self.assertTrue(all(r['pairs']==2 for r in common['paired']))


if __name__=='__main__':unittest.main()
