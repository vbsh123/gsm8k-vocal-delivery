from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from audit_delivery_results import audit
from compare_cloud_accuracy import PROMPT
from scoring import score_response


class DeliveryAuditTests(unittest.TestCase):
    def fixture(self, root, number=1):
        folder=root/str(number);folder.mkdir()
        task=dict(provider='azure',audio_path='n.wav',question_id=f'q{number}',voice='Jenny',style='neutral',answer='18')
        settings=dict(seed=42,prompt=PROMPT,model='test',max_new_tokens=512)
        row=dict(task,**settings,response='Final answer: 18',**score_response('Final answer: 18','18'))
        (folder/'plan.jsonl').write_text(json.dumps(task)+'\n')
        (folder/'qwen-omni-settings.json').write_text(json.dumps(settings))
        (folder/'results-qwen-omni.jsonl').write_text(json.dumps(row)+'\n')
        return folder,row

    def test_disjoint_cohorts_with_identical_filenames_are_both_counted(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            a,_=self.fixture(Path(d),1);b,_=self.fixture(Path(d),2)
            self.assertEqual(audit([a,b],'qwen-omni'),2)

    def test_rejects_wrong_reference_seed_score_and_duplicate(self):
        for field,value in [('reference_answer','19'),('seed',7),('correct',False),('style','angry')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                folder,row=self.fixture(Path(d));row[field]=value
                (folder/'results-qwen-omni.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaises(ValueError):audit([folder],'qwen-omni')
        with tempfile.TemporaryDirectory() as d:
            folder,row=self.fixture(Path(d))
            (folder/'results-qwen-omni.jsonl').write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'duplicate answer'):audit([folder],'qwen-omni')

    def test_rejects_overlap_between_cohorts(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);folder,_=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'multiple folders'):audit([folder,folder],'qwen-omni')


if __name__=='__main__':unittest.main()
