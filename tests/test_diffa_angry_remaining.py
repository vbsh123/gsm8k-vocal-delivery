import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import diffa_angry_remaining as run
from tests.test_diffa_bbh_angry125 import fixture


class RemainingTests(unittest.TestCase):
    def test_generalized_half_selection_and_ifeval_budget(self):
        data,labels=fixture()
        ids=data['collection_ids'][:3]+data['test_ids'][:3]
        data['collection_ids']=ids[:3];data['test_ids']=ids[3:]
        data['pinned']={k:v for k,v in data['pinned'].items() if k.split('|')[0] in ids}
        data['target'].update(name='ifeval',task='ifeval',max_new_tokens=1024)
        for row in labels.values():row['emotion']='angry'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cohorts.json').write_text('{}')
            with patch.object(run.half,'collection_bank',side_effect=lambda p:p['bank']):
                plan=run.half.prepare('ifeval',data,labels,root,root/'new',half=3)
            self.assertEqual(len(plan['pairs']),3)
            self.assertEqual(len(plan['test_ids']),3)
            self.assertEqual(plan['max_new_tokens'],1024)
            self.assertFalse(set(plan['collection_ids'])&set(plan['test_ids']))
            destination=root/'eval';destination.mkdir()
            record=dict(question_id='new',voice='Jenny',reference_answer='')
            with patch.object(run.collection,'generate',return_value=('response',{})):
                run.half.fill_neutral(plan,dict(target=dict(records=[record])),destination,None,
                                      SimpleNamespace(encode=lambda *a,**k:[1]),None)
            self.assertEqual(run.half.read_rows(destination/'results.jsonl')[0]['max_new_tokens'],1024)

    def test_svamp_493_uses_only_original_collection_and_keeps_500_test(self):
        train=[f'c{i}' for i in range(500)];test=[f't{i}' for i in range(500)]
        pinned={};labels={}
        for i,q in enumerate(train):
            for style in ('neutral','angry'):
                pinned[q+'|'+style]=dict(question='question '+q,record=dict(question_id=q,
                    voice='Jenny',audio_sha256=q+style,reference_answer='2'))
            labels[q+'angry']=dict(classification_status='classified',emotion='angry' if i>=7 else 'neutral')
        original=dict(run='original',records=[dict(question_id=q,reference_answer='2') for q in test],
                      bank={q:dict(question='question '+q,answer='2') for q in test},max_new_tokens=512,
                      prompt='prompt',vector_info=dict(collection_prompt='prompt'))
        data=dict(target=original,collection_ids=train,test_ids=test,pinned=pinned)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cohorts.json').write_text('{}')
            plan=run.prepare_svamp(data,labels,root,root/'new')
            self.assertEqual(len(plan['pairs']),493)
            self.assertEqual(plan['test_ids'],test)
            self.assertEqual(plan['target']['records'],original['records'])
            self.assertEqual(plan['common_test_ids'],test)
            self.assertEqual(set(plan['collection_ids']),set(train[7:]))
            self.assertFalse(set(plan['collection_ids'])&set(test))
            self.assertEqual(len(plan['classifier_receipts']),493)
            self.assertEqual(original['records'],plan['original_target']['records'])

    def test_existing_audit_can_be_reused_without_new_classification(self):
        with tempfile.TemporaryDirectory() as d:
            previous=Path.cwd()
            try:
                os.chdir(d);aroot=Path('outputs/diffa-angry-agreed/audit');aroot.mkdir(parents=True)
                (aroot/'cohorts.json').write_text(json.dumps(dict(datasets={'svamp':{}})))
                source=Path('source.json');source.write_text('{}');vector=Path('original.pt');vector.write_bytes(b'vector')
                target=dict(config_file=str(source),config_sha256=run.audit.winners.digest(source),
                            vector=str(vector),vector_sha256=run.audit.winners.digest(vector))
                prepared=dict(original_target=target,target=target)
                args=SimpleNamespace(output=Path('new'),mode='prepare',datasets=['svamp'])
                with patch.object(run,'prepare_svamp',return_value=prepared) as prepare, \
                     patch.object(run.audit.winners,'score_rows'),patch.object(run.subprocess,'run') as classify:
                    self.assertEqual(run.audit_and_prepare(args),{'svamp':prepared})
                self.assertEqual(prepare.call_args.args[2],aroot)
                classify.assert_not_called()
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
