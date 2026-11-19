"""IFEval must preserve its cohort, token budget and official scoring path."""
from contextlib import redirect_stdout
import io,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phi4_ifeval as ifeval
import phi4_task_steering as task
from kimi_bbh_layer_diagnostic import freeze


class PhiIFEvalTests(unittest.TestCase):
    def test_frozen508_split_has254_disjoint_questions_each(self):
        split=json.loads(ifeval.SPLIT.read_text())
        self.assertEqual(len(split['collection_ids']),254);self.assertEqual(len(split['test_ids']),254)
        self.assertEqual(len(set(split['collection_ids']+split['test_ids'])),508)
        self.assertEqual(split['official_revision'],ifeval.official.REVISION)

    def test_native_generation_uses1024_only_for_ifeval_plan(self):
        data=dict(task='ifeval',max_new_tokens=1024,collection=[],test=[dict(question_id='q',neutral=dict(question_id='q',voice='Jenny'),angry=dict(question_id='q',voice='Jenny'))],bank={'q':dict(prompt='Follow this')},prompt=ifeval.official.PROMPT)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json';freeze(path,data);plan=task.make_plan(data,path,17,'answer',-1.)
            self.assertEqual(plan['max_new_tokens'],1024)
            with patch.object(task.engine,'NativePhi'),patch.object(task.native_hooks,'generate',return_value={'response':'Answer'}) as generate:
                task.Native(plan).generate({},plan['prompt'])
            self.assertEqual(generate.call_args.kwargs['max_new_tokens'],1024)

    def test_official_prompt_checks_and_instruction_totals(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            output=Path(tmp);labels=['neutral','angry','neutral-ifeval-L17-answer-a-1']
            plan=dict(task='ifeval',layer=17,group='answer',strength=-1.,bank={'q':{}},recordings=[dict(condition=c) for c in labels])
            freeze(output/'run_config.json',plan)
            rows=[dict(question_id='q',voice='Jenny',condition=c,response='bad' if c=='neutral' else 'good',response_tokens=3) for c in labels]
            def score(example,response,evaluator):
                passed=response=='good'
                return {mode:dict(prompt=passed,instructions=[True,passed]) for mode in ('strict','loose')}
            with patch.object(task,'completed',return_value=({},dict(enumerate(rows)))),patch.object(ifeval.official,'setup_official'),patch.object(ifeval.official,'official_scorer'),patch.object(ifeval.official,'score',side_effect=score):
                task.report(output)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['primary_metric'],'strict_prompt')
            self.assertEqual(summary['totals'][0]['strict_instruction'],.5)
            self.assertEqual(summary['totals'][2]['strict_prompt'],1.)
            self.assertEqual(summary['paired'][0]['gain_pp'],100.)
            self.assertEqual(len((output/'official-scored.jsonl').read_text().splitlines()),3)


if __name__=='__main__':unittest.main()
