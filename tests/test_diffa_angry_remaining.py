import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
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

    def test_phi_export_packages_exact_existing_500_and_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            previous=Path.cwd()
            try:
                os.chdir(d);folder=Path('outputs/phi4-next-tasks/svamp');source=folder/'audio';source.mkdir(parents=True)
                rows=[];pairs=[]
                for i in range(500):
                    q=f'svamp_chal-{i}';path=source/f'{q}.wav';path.write_bytes(str(i).encode())
                    rows.append(dict(question_id=q,question=q,answer='2',style='angry',provider='azure',voice='en-US-JennyNeural',audio_path=path.name))
                    pairs.append(dict(question_id=q,angry=dict(audio_file=str(path.absolute()),audio_sha256=run.audit.winners.digest(path))))
                (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (folder/'data_config.json').write_text(json.dumps(dict(source=str(source.absolute()),test=pairs)))
                archive=Path('export.zip');run.export_svamp(archive)
                with zipfile.ZipFile(archive) as z:
                    exported=[json.loads(l) for l in z.read('plan.jsonl').decode().splitlines()]
                    self.assertEqual(len(exported),500)
                    self.assertEqual(len(json.loads(z.read('receipts.json'))['recordings']),500)
                    self.assertTrue(all(r['style']=='angry' for r in exported))
                    self.assertEqual(z.read(exported[0]['audio_path']),b'0')
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
