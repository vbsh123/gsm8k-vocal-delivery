from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import diffa_happy_nightly as run
import diffa_collection_report as report
import diffa_stream_collection as stream


class CollectionReportTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'gsm8k';folder.mkdir()
        pairs=[dict(question_id=q,**{key:dict(question_id=q,voice='Jenny',style=style,answer='2',reference_answer='2',audio_file=str(root/'audio/clip.wav'))
            for key,style in (('neutral','neutral'),('emotion','happy'))}) for q in ('c0','c1')]
        plan=dict(name='gsm8k',emotion='happy',pairs=pairs,collection_ids=['c0','c1'],test_ids=['heldout'],
            bank={q:dict(question='Question '+q,answer='2') for q in ('c0','c1','heldout')},prompt='heldout prompt',collection_prompt='collection prompt',
            max_new_tokens=512,target=dict(task='gsm8k',bank={'heldout':dict(answer='2')}))
        (folder/'run_config.json').write_text(json.dumps(plan));sha=run.winners.digest(folder/'run_config.json')
        state=stream.load(plan,folder,sha)
        for i,pair in enumerate(pairs):
            saved=dict(question_id=pair['question_id'],neutral=torch.ones(32,4,2),emotion=torch.full((32,4,2),2.),difference=torch.ones(32,4,2),
                neutral_response='#### '+('2' if i==0 else '1'),emotion_response='#### 2',neutral_forwards=3,emotion_forwards=2,matched_forwards=2,
                neutral_encoding={},emotion_encoding={})
            stream.append(state,saved)
        run.save_tensor(torch,state,folder/'collection-stream.pt')
        delivery=folder/'delivery';delivery.mkdir();(delivery/'results.jsonl').write_text(json.dumps(dict(question_id='heldout',voice='Jenny',condition='neutral',response='#### 2'))+'\n')
        return plan,folder

    def test_saved_collection_responses_are_scored_without_new_generation_or_test_pooling(self):
        with tempfile.TemporaryDirectory() as d:
            plan,folder=self.fixture(Path(d));checkpoint=run.winners.digest(folder/'collection-stream.pt');baseline=run.winners.digest(folder/'delivery/results.jsonl')
            tokenizer=SimpleNamespace(encode=lambda text,**kw:text.split())
            with patch.object(run,'generate',side_effect=AssertionError('no new generations')),redirect_stdout(io.StringIO()):run.report_collection(plan,folder,tokenizer)
            summary=json.loads((folder/'collection-report/summary.json').read_text())
            self.assertEqual(summary['cohort'],'collection');self.assertEqual(summary['matched_questions'],2)
            self.assertEqual(summary['expected_questions'],2);self.assertEqual(summary['prompt'],'collection prompt')
            result=next(r for r in summary['paired'] if r['metric']=='flexible-numeric')
            self.assertEqual((result['baseline_correct'],result['steered_correct']),(1,2))
            rows=run.read_rows(folder/'collection-report/results.jsonl')
            self.assertEqual({r['question_id'] for r in rows},{'c0','c1'})
            self.assertTrue(all(r['split']=='collection' and r['prompt']=='collection prompt' for r in rows))
            self.assertEqual(checkpoint,run.winners.digest(folder/'collection-stream.pt'))
            self.assertEqual(baseline,run.winners.digest(folder/'delivery/results.jsonl'))

    def test_partial_checkpoint_reports_only_committed_matched_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            plan,folder=self.fixture(Path(d));path=folder/'collection-stream.pt';state=torch.load(path,weights_only=True)
            state['completed_ids']=state['completed_ids'][:1];state['rows']=state['rows'][:1];torch.save(state,path)
            rows=report.collection_rows(plan,folder)
            self.assertEqual([(r['question_id'],r['condition']) for r in rows],[('c0','neutral'),('c0','happy')])

    def test_bbh_collection_recovers_original_choice_metadata_and_checks_text(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan,folder=self.fixture(root);audio=root/'audio';audio.mkdir()
            plan['name']='bbh3';plan['target']['task']='bbh3'
            questions=[]
            for qid in plan['collection_ids']:
                plan['bank'][qid]['answer']='A'
                questions.append(dict(question_id=qid,question=plan['bank'][qid]['question'],answer='A',original_input='Which?\nOptions:\n(A) A\n(B) B\n(C) C',original_target='(A)'))
            (audio/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
            bank=report.collection_bank(plan)
            self.assertEqual(bank['c0']['original_target'],'(A)');self.assertNotIn('original_target',plan['bank']['c0'])
            questions[0]['question']='wrong text';(audio/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
            with self.assertRaises(ValueError):report.collection_bank(plan)

    def test_ifeval_collection_uses_official_instruction_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();example=dict(key=17,prompt='Write two words.',instruction_id_list=['length_constraints:number_words'],kwargs=[dict(num_words=2,relation='at least')])
            (data/'input_data.jsonl').write_text(json.dumps(example)+'\n')
            plan=dict(name='ifeval',collection_ids=['ifeval_17'],bank={'ifeval_17':dict(question='Write two words.',answer='')})
            with patch('run_spoken_ifeval.setup_official',return_value=root):bank=report.collection_bank(plan)
            self.assertEqual(bank['ifeval_17']['instruction_id_list'],example['instruction_id_list'])
            self.assertEqual(bank['ifeval_17']['kwargs'],example['kwargs'])

    def test_unmatched_heldout_baseline_is_not_reported_as_collection_failure(self):
        plan=dict(name='gsm8k',test_ids=['heldout'])
        rows=[dict(question_id='heldout',voice='Jenny',condition='neutral')]
        with tempfile.TemporaryDirectory() as d,patch.object(run.winners,'score_rows',side_effect=AssertionError('no pairs to score')),redirect_stdout(io.StringIO()) as printed:
            run.report_delivery(plan,Path(d),rows=rows)
        self.assertIn('matched held-out',printed.getvalue());self.assertIn('neutral=1 happy=0 saved',printed.getvalue())

if __name__=='__main__':unittest.main()
