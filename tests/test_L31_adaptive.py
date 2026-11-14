from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from run_L31_adaptive import fit_mapping,predict_raw,predict_delta,report


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class AdaptiveMappingTests(unittest.TestCase):
    def test_predicts_state_dependent_corrections_and_weights_questions_equally(self):
        import torch
        torch.manual_seed(42)
        xs=[torch.randn(10,4)+i*.2 for i in range(20)]
        transform=torch.randn(4,4)
        rows=[dict(neutral=x,difference=x@transform+2) for x in xs]
        model=fit_mapping(rows,torch,rank=4,ridge=.001)
        test=torch.randn(20,4)
        self.assertLess(float((predict_raw(model,test)-(test@transform+2)).square().mean()),.01)
        repeated=[dict(neutral=r['neutral'].repeat(3,1),difference=r['difference'].repeat(3,1)) if i==0 else r
                  for i,r in enumerate(rows)]
        other=fit_mapping(repeated,torch,rank=4,ridge=.001)
        torch.testing.assert_close(predict_raw(model,test),predict_raw(other,test),atol=1e-4,rtol=1e-4)
        self.assertFalse(torch.equal(predict_raw(model,test[0]),predict_raw(model,test[1])))

    def test_same_state_mean_has_incumbent_norm_and_adaptive_delta_is_bounded(self):
        import torch
        rows=[dict(neutral=torch.ones(4,3)*i,difference=torch.ones(4,3)*(i+1)) for i in range(5)]
        model=fit_mapping(rows,torch,rank=2,ridge=1)
        model.update(reference_norm=3.,gain=3/float(model['mean'].norm()))
        mean=predict_delta(model,torch.ones(3),False,1.)
        self.assertAlmostEqual(float(mean.norm()),3.,places=5)
        full=predict_delta(model,torch.ones(3)*1e9,True,1.)
        half=predict_delta(model,torch.ones(3)*1e9,True,.5)
        self.assertLessEqual(float(full.norm()),6.00001)
        torch.testing.assert_close(half,full*.5)


class AdaptiveReportTests(unittest.TestCase):
    def test_pools_disjoint_cohorts_and_includes_incumbent(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);output=root/'out';output.mkdir();sources=[]
            for name,qid,bc,sc in [('b','q1',False,True),('c','q2',True,False)]:
                source=root/name;source.mkdir();sources.append((name,source))
                (source/'run_config.json').write_text(json.dumps(dict(recordings=[dict(question_id=qid)])))
                rows=[dict(clip_id=qid,condition='baseline',correct=bc),dict(clip_id=qid,condition='L31-answer-a1',correct=sc)]
                (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=report(output,sources)
            row=next(r for r in result if r['condition']=='original-L31-a1')
            self.assertEqual((row['pairs'],row['wins'],row['losses']),(2,1,1))


if __name__=='__main__':unittest.main()
