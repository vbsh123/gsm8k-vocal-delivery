from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_L31_timing import conditions,main,report


class L31TimingTests(unittest.TestCase):
    def test_only_requested_schedules_on_both_exact_source_cohorts(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);vector=root/'vector.pt';vector.touch()
            for name in ('original_mean','third-cohort/original_mean'):
                source=root/name;source.mkdir(parents=True)
                (source/'run_config.json').write_text(json.dumps(dict(settings=dict(vectors=str(vector)),recordings=[dict(question_id=name)])))
            with patch('sys.argv',['test','--late-root',str(root),'--output',str(root/'out'),'--prepare-only']),\
                 patch('run_L31_timing.subprocess.run') as run:
                main()
            commands=[c.args[0] for c in run.call_args_list]
            self.assertEqual(len(commands),2)
            self.assertTrue(all(c[-1]=='--prepare-only' for c in commands))
            self.assertEqual([c[c.index('--source-run')+1] for c in commands],[str(root/'original_mean'),str(root/'third-cohort/original_mean')])
            self.assertTrue(all(c[c.index('--baseline-source')+1]==c[c.index('--source-run')+1] for c in commands))
            self.assertEqual([r.get('schedule') for r in conditions()[1:]],['early','middle','late'])
            self.assertTrue(all(r['interventions']==[dict(layer=31,group='answer',strength=1)] for r in conditions()[1:]))

    def test_pooled_summary_keeps_saved_winner_and_counts_only_paired_new_answers(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'out';out.mkdir();sources=[]
            for name,qid,bc,sc in [('second','b',False,True),('third','c',True,False)]:
                source=root/name;source.mkdir();sources.append((name,source))
                (source/'run_config.json').write_text(json.dumps(dict(recordings=[dict(question_id=qid,clip_id=qid)])))
                baseline=dict(clip_id=qid,condition='baseline',correct=bc)
                saved=[baseline,dict(clip_id=qid,condition='L31-answer-a1',correct=sc)]
                (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
                dest=out/name/'original_mean';dest.mkdir(parents=True)
                new=[baseline,dict(clip_id=qid,condition='L31-answer-a1-early',correct=sc)]
                (dest/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
            summary=report(out,sources)
            rows={r['condition']:r for r in summary}
            self.assertEqual((rows['L31-answer-a1']['pairs'],rows['L31-answer-a1']['wins'],rows['L31-answer-a1']['losses']),(2,1,1))
            self.assertEqual(rows['L31-answer-a1-early']['pairs'],2)


if __name__=='__main__':unittest.main()
