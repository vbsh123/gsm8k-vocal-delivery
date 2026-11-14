from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate_arc_audio import PROMPT,STEERING
from generate_arc_azure import select_questions
from steer_arc_mean import ARC_STEERING,CS_STEERING,run,split_recordings,summary


class ArcMeanTests(unittest.TestCase):
    def test_250_train_and_250_test_disjoint_and_ignore_transfer_rows(self):
        records=[dict(question_id=f'q{i:03}',voice='Jenny',condition=c,style='neutral' if c==STEERING else c)
            for i in range(500) for c in ('neutral','angry','happy','sad',STEERING)]
        train,test=split_recordings(records)
        self.assertEqual((len(train),len(test)),(250,250))
        self.assertFalse({r['question_id'] for r in train}&{r['question_id'] for r in test})
        self.assertEqual((train,test),split_recordings(records))
        self.assertTrue(all(r['condition']=='neutral' for r in test))
        self.assertTrue(all(r['angry']['condition']=='angry' for r in train))

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Needs NumPy for confidence intervals')
    def test_shared_report_compares_both_vectors_on_test_questions_only(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'out';out.mkdir();bench=root/'bench';bench.mkdir()
            def question(name):return dict(question_id=name,answer='A',choices=[dict(label=k,text=v) for k,v in zip('ABCD',['water','oil','sand','air'])])
            (root/'questions.jsonl').write_text('\n'.join(json.dumps(question(q)) for q in ('train','test'))+'\n')
            (out/'split.json').write_text(json.dumps(dict(extraction=[dict(question_id='train')],test=[dict(question_id='test',voice='Jenny')])))
            def row(q,c,r):return dict(question_id=q,voice='Jenny',condition=c,response=r,reference_answer='A')
            rows=[row('test','neutral','A'),row('test',ARC_STEERING,'Final answer: A')]
            (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            transfer=[row('train',STEERING,'Final answer: A'),row('test',STEERING,'Final answer: B')]
            (bench/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in transfer))
            summary(out,root,bench)
            report=json.loads((out/'shared-heldout-summary.json').read_text())
            self.assertEqual({r['condition']:r['n'] for r in report['accuracy']},{'neutral':1,ARC_STEERING:1,CS_STEERING:1})


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class ArcMeanRuntimeTests(unittest.TestCase):
    def test_matched_forward_collection_frozen_seed_and_heldout_only_steering(self):
        import numpy as np
        import torch
        class Block(torch.nn.Module):
            def forward(self,h):return (h,None)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        model=SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        calls=[]
        def generate(audio,sr,model,tokenizer,features,prompt,limit):
            for _ in range(3 if audio[0]==3 else 2):
                h=torch.ones(1,520,4)*float(audio[0])
                for layer in blocks:h=layer(h)[0]
            calls.append((float(h[0,0,0]),float(h[0,-1,0]),float(torch.rand(())),prompt,limit))
            return 'Final answer: A'
        original_to=torch.Tensor.to
        def cpu_to(tensor,*a,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return original_to(tensor,*a,**kw)
        load=SimpleNamespace(load_model=lambda:(model,None,None),gen=generate)
        modules={'soundfile':SimpleNamespace(read=lambda path,**kw:(np.array([3 if 'angry' in str(path) else 1],dtype=np.float32),16000)),
            'extract_vectors':load}
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'out';out.mkdir();bench=root/'bench';bench.mkdir()
            def task(q,style):
                file=root/f'{q}-{style}.wav';file.write_bytes(b'wave');stat=file.stat()
                return dict(question_id=q,voice='Jenny',style=style,condition=style,audio_path=file.name,audio_file=str(file),
                    audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns,answer='A')
            n=task('train','neutral');a=task('train','angry');test=task('test','neutral')
            plan=dict(extraction=[dict(question_id='train',voice='Jenny',neutral=n,angry=a)],test=[test],averaging='matched')
            questions=[dict(question_id=q,answer='A',choices=[dict(label=k,text=k) for k in 'ABCD']) for q in ('train','test')]
            (root/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in questions))
            cached=[dict(question_id=t['question_id'],voice='Jenny',condition=t['style'],response='Final answer: A',
                reference_answer='A',seed=42,prompt=PROMPT) for t in (n,a,test)]
            (bench/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cached))
            with patch.dict('sys.modules',modules),patch.object(torch.Tensor,'to',cpu_to):
                run(plan,root,bench,out);run(plan,root,bench,out)
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[-1][:2],(1.,3.))
            self.assertTrue(all(c[2:]==calls[0][2:] for c in calls))
            self.assertEqual(calls[0][3:],(PROMPT,512))
            pair=torch.load(out/'pairs/00000.pt',weights_only=True)
            self.assertEqual((pair['neutral_forwards'],pair['angry_forwards'],pair['matched_forwards']),(2,3,2))
            self.assertTrue(torch.equal(pair['difference'],torch.full((4,),2.)))
            vector=torch.load(out/'style-vectors.pt',weights_only=True)
            self.assertEqual(vector['extraction_question_ids'],['train']);self.assertEqual(vector['collected_targets'],[(31,'answer')])
            results=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
            self.assertEqual([r['question_id'] for r in results],['test','test'])
            self.assertEqual([r['condition'] for r in results],['neutral',ARC_STEERING])
            self.assertTrue(all(not b._forward_hooks for b in blocks))


if __name__=='__main__':unittest.main()
