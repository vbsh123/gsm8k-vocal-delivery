from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diffa_bbh_angry125 as run


def fixture():
    ids=[f'q{i:03d}' for i in range(250)];pinned={};labels={}
    for i,q in enumerate(ids):
        for style in ('neutral','angry'):
            key=q+style
            pinned[q+'|'+style]=dict(question='question '+q,record=dict(question_id=q,voice='Jenny',
                audio_sha256=key,audio_file='/audio/'+key+'.wav',reference_answer='A'))
        # Only100 of the original125 training clips are recognized; neutral has NO labels.
        labels[q+'angry']=dict(classification_status='classified',emotion='angry' if i%5 else 'neutral')
    target=dict(name='bbh3',task='bbh3',run='/original',prompt='prompt',vector_info=dict(collection_prompt='prompt'),
                bank={q:dict(question='question '+q,answer='A',original_input='input',original_target='(A)') for q in ids[125:]})
    return dict(collection_ids=ids[:125],test_ids=ids[125:],pinned=pinned,target=target),labels


class BBHAngry125Tests(unittest.TestCase):
    def test_exact_125_angry_only_selection_and_disjoint_remainder(self):
        data,labels=fixture();before=deepcopy(data)
        pairs,train,test=run.select(data,labels)
        self.assertEqual((len(train),len(test)),(125,125))
        self.assertFalse(set(train)&set(test))
        self.assertEqual(set(train+test),set(data['collection_ids']+data['test_ids']))
        self.assertTrue(all(labels[pairs[q]['emotion']['audio_sha256']]['emotion']=='angry' for q in train))
        self.assertEqual(len(set(train)&set(data['collection_ids'])),100)
        self.assertEqual(len(set(test)&set(data['test_ids'])),100)
        self.assertEqual(run.select(data,labels)[1:],(train,test))
        self.assertEqual(data,before)

    def test_fails_before_collection_if_not_enough_recognized_or_labels_pending(self):
        data,labels=fixture()
        labels['q001angry']['classification_status']='error'
        with self.assertRaisesRegex(ValueError,'Complete angry classification'):run.select(data,labels)
        for i,row in enumerate(labels.values()):
            row['classification_status']='classified';row['emotion']='angry' if i<124 else 'neutral'
        with self.assertRaisesRegex(ValueError,'need125'):run.select(data,labels)

    def test_plan_records_new_split_and_preserves_original(self):
        data,labels=fixture();original=deepcopy(data['target'])
        def bank(plan):
            return {q:dict(row,original_input='input',original_target='(A)') for q,row in plan['bank'].items()}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cohorts.json').write_text('{}')
            with patch.object(run,'collection_bank',side_effect=bank):
                plan=run.prepare('bbh3',data,labels,root,root/'new')
            self.assertEqual(data['target'],original)
            self.assertEqual(plan['original_target'],original)
            self.assertEqual(len(plan['target']['records']),125)
            self.assertEqual(set(plan['target']['bank']),set(plan['test_ids']))
            self.assertEqual(len(plan['common_test_ids']),100)
            self.assertFalse(set(plan['collection_ids'])&set(plan['test_ids']))
            self.assertEqual(plan['max_new_tokens'],512)
            self.assertEqual(plan['prompt'],plan['collection_prompt'])
            self.assertTrue(all(set(r)=={'question_id','angry'} for r in plan['classifier_receipts']))

    def test_only_missing_new_test_baselines_generated(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'results.jsonl').write_text(json.dumps(dict(question_id='old',condition='neutral'))+'\n')
            records=[dict(question_id=q,voice='Jenny',reference_answer='A') for q in ('old','new')]
            tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2])
            with patch.object(run.collection,'generate',return_value=('The answer is (A).',{})) as generate:
                run.fill_neutral(dict(name='bbh3',prompt='prompt'),dict(target=dict(records=records)),folder,None,tokenizer,None)
                run.fill_neutral(dict(name='bbh3',prompt='prompt'),dict(target=dict(records=records)),folder,None,tokenizer,None)
            self.assertEqual(generate.call_count,1)
            self.assertEqual(generate.call_args.args[0]['question_id'],'new')

    def test_direct_comparison_uses_only_questions_held_out_from_both_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';old.mkdir();new=root/'new';new.mkdir()
            (old/'run_config.json').write_text('{}')
            (old/'results.jsonl').write_text(''.join(json.dumps(dict(question_id=q,strength=1.,response='wrong'))+'\n'
                                                   for q in ['common','not-common']))
            (new/'results.jsonl').write_text(''.join(json.dumps(dict(question_id=q,strength=a,response='right'))+'\n'
                                                   for q in ['common','not-common'] for a in [0.,1.]))
            plan=dict(name='bbh3',original_target=dict(run=str(old)),common_test_ids=['common'],
                      target=dict(records=[dict(question_id='common'),dict(question_id='not-common')]))
            def score(job,rows,folder):return [dict(r,metrics={'flexible-extract':r['response']=='right'}) for r in rows],{}
            with patch.object(run.agreed.winners,'saved_strength',side_effect=lambda r,*args:r['strength']), \
                 patch.object(run.agreed.winners,'reusable',return_value=True), \
                 patch.object(run.agreed.winners,'score_rows',side_effect=score):
                run.compare_original(plan,new)
            summary=json.loads((new/'common-heldout/summary.json').read_text())
            self.assertEqual(summary['question_ids'],['common'])
            self.assertEqual(summary['paired'][0]['gain_pp'],100.)


if __name__=='__main__':unittest.main()
