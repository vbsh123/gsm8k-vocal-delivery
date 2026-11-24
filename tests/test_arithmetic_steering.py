from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
import run_arithmetic_steering as run


def parent_fixture(root):
    parent = root/'pilot'; parent.mkdir()
    ids = [f'bbh_arithmetic_{i:03d}' for i in range(50)]
    bank = {q:dict(question_id=q, original_input='((-1 + 2) * (3 - 4)) =',
        original_target='-1', question='Spoken arithmetic question', answer='-1') for q in ids}
    records = []
    for q in ids:
        for c in run.pilot.CONDITIONS:
            style = c.removesuffix('-reasoning')
            path = root/(q+'-'+style+'.wav'); path.write_bytes((q+style).encode())
            records.append(dict(question_id=q, condition=c, style=style, voice=run.pilot.base.VOICE,
                audio_file=str(path), audio_sha256=run.winners.digest(path), reference_answer='-1'))
    plan = dict(protocol=run.pilot.PROTOCOL, name=run.pilot.TASK, model='diffa',
        source_git_blob=run.pilot.SOURCE_BLOB, prompts=run.pilot.PROMPTS,
        recipe=run.night.vanilla.RECIPE, question_order=ids, bank=bank, records=records)
    config = parent/'run_config.json'; config.write_text(json.dumps(plan))
    (parent/'runtime.json').write_text(json.dumps(dict(model_revisions={},packages={})))
    rows = [dict(r, prompt=plan['prompts'][r['condition']], response='The answer is -1.',
        response_tokens=7, model='diffa', strength=0., interventions=[],
        run_config_sha256=run.winners.digest(config), **plan['recipe']) for r in records]
    (parent/'deliveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return parent


class Backend:
    def __init__(self, stop_after=None, wrong=False):
        self.calls=[]; self.stop_after=stop_after; self.wrong=wrong
    def generate(self, record, plan, capture=False):
        if self.stop_after is not None and len(self.calls)>=self.stop_after: raise KeyboardInterrupt
        self.calls.append((record['question_id'],record['condition'],capture,plan['prompt']))
        value = 2. if record['style']=='neutral' else 7.
        return dict(response='The answer is 0.' if self.wrong else 'The answer is -1.',
            response_tokens=7, audio_encoding={}, forwards=2,
            full=torch.full((32,2,4,3),value*10), active=torch.full((32,2,3),value))


class ArithmeticSteeringTests(unittest.TestCase):
    def test_split_capture_resume_active_block_vector_and_official_heldout_scoring(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory); parent=parent_fixture(root); output=root/'steering'
            original=(parent/'deliveries.jsonl').read_bytes()
            experiment,p,old,rows=run.prepare(parent,output)
            self.assertEqual(len(experiment['collection_ids']),25)
            self.assertEqual(len(experiment['evaluation_ids']),25)
            self.assertFalse(set(experiment['collection_ids']) & set(experiment['evaluation_ids']))
            self.assertEqual(set(experiment['collection_ids'])|set(experiment['evaluation_ids']),set(old['question_order']))
            self.assertEqual(p['prompt'],run.pilot.BARE_PROMPT)
            run.prepare(parent,output)
            backend=Backend(stop_after=2)
            with self.assertRaises(KeyboardInterrupt):run.collect(p,output/'collection',rows,backend)
            self.assertEqual(len(run.night.load_state(p,output/'collection')['completed_ids']),1)
            run.report(output)
            backend.stop_after=None
            run.collect(p,output/'collection',rows,backend)
            run.collect(p,output/'collection',rows,backend)
            self.assertEqual(len(backend.calls),50)
            self.assertTrue(all(capture and prompt==run.pilot.BARE_PROMPT for _,_,capture,prompt in backend.calls))
            self.assertEqual({q for q,_,_,_ in backend.calls},set(experiment['collection_ids']))
            vector,info=run.winners.read_vector(output/'collection/style-vectors.pt','angry')
            torch.testing.assert_close(vector,torch.full((3,),5.))
            self.assertEqual(info['pairs'],25)
            self.assertEqual(set(info['extraction_ids']),set(experiment['collection_ids']))
            folder,job=run.steering_job(experiment,p,old,rows,output)
            self.assertEqual(job['strengths'],[0.,2.])
            self.assertEqual({r['question_id'] for r in job['target']['records']},set(experiment['evaluation_ids']))
            baseline=run.follow.validate(job,folder)
            self.assertEqual(len(baseline),25)
            run.steering_job(experiment,p,old,rows,output)
            self.assertEqual(len(run.follow.validate(job,folder)),25)
            with (folder/'results.jsonl').open('a') as stream:
                for i,row in enumerate(baseline.values()):
                    stream.write(json.dumps(dict(row,strength=2.,condition='L31-answer-a2',
                        vector_sha256=job['source']['vector_sha256'],response='The answer is -1.' if i%2 else 'The answer is 0.',
                        interventions=[dict(layer=31,group='answer',strength=2.)]))+'\n')
            run.report(output)
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['scorer']['task'],'bbh_cot_zeroshot_multistep_arithmetic_two')
            paired=[r for r in summary['paired'] if r['strength']==2.]
            self.assertEqual({r['metric'] for r in paired},{'strict-match','flexible-extract'})
            self.assertTrue(all(r['pairs']==25 and r['baseline_correct']==25 and r['steered_correct']==12 for r in paired))
            self.assertEqual(summary['lengths']['L31-answer-a2']['n'],25)
            self.assertEqual((parent/'deliveries.jsonl').read_bytes(),original)

    def test_changed_replay_is_not_committed(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory); parent=parent_fixture(root); output=root/'steering'
            _,p,_,rows=run.prepare(parent,output)
            with self.assertRaisesRegex(ValueError,'differs from its saved bare response'):
                run.collect(p,output/'collection',rows,Backend(wrong=True))
            self.assertEqual(run.night.load_state(p,output/'collection')['completed_ids'],[])
            self.assertFalse((output/'collection/style-vectors.pt').exists())


if __name__=='__main__':unittest.main()
