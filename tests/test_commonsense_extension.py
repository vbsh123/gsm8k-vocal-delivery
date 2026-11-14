from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from extend_commonsense_steering import select_records,run
from generate_commonsense_azure import questions
from rescore_audio_steering import report,harness_filter


class ExtensionTests(unittest.TestCase):
    def test_source_slice_preserves_global_indices_and_has_no_overlap(self):
        rows=[dict(id=str(i),answerKey='A',question=dict(stem='Question?',choices=[dict(label=k,text=k) for k in 'ABCDE'])) for i in range(1221)]
        data='\n'.join(json.dumps(r) for r in rows).encode()
        first=questions(data,500);rest=questions(data,721,500)
        self.assertEqual([r['source_index'] for r in rest],list(range(500,1221)))
        self.assertFalse({r['question_id'] for r in first}&{r['question_id'] for r in rest})

    def test_new_questions_cannot_include_extraction_or_previous_test(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with wave.open(str(root/'neutral.wav'),'wb') as audio:
                audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(b'\0\0'*160)
            task=dict(provider='azure',style='neutral',question_id='new',audio_path='neutral.wav')
            (root/'plan.jsonl').write_text(json.dumps(task)+'\n')
            self.assertEqual(len(select_records(root,dict(extraction=[],test=[]))),1)
            for field in ('extraction','test'):
                split=dict(extraction=[],test=[]);split[field]=[dict(question_id='new')]
                with self.assertRaisesRegex(ValueError,'overlap'):select_records(root,split)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Needs NumPy for confidence intervals')
    def test_upstream_filter_loaded_and_paired_rescoring_exposes_format_gain(self):
        source='''import re
@register_filter("regex")
class RegexFilter(Filter):
    def __init__(self, regex_pattern, group_select=0):
        self.regex=re.compile(regex_pattern);self.group_select=group_select
    def apply(self,resps,docs):
        def extract(s):
            m=self.regex.findall(s)
            if not m:return "[invalid]"
            v=m[self.group_select]
            return next(x for x in v if x) if isinstance(v,tuple) else v
        return [[extract(s) for s in r] for r in resps]
'''
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);(root/'revision.txt').write_text('fixture')
            (root/'extraction.py').write_text(source)
            (root/'gsm8k.yaml').write_text(r'(-?[$0-9.,]{2,})|(-?[0-9]+)')
            with patch('rescore_audio_steering.download',side_effect=AssertionError('No download expected')):
                Filter=harness_filter(root)
            bank=[dict(answer='Worked solution #### 42')]
            rows=[dict(question_id='gsm8k_test_00000',condition=c,reference_answer='42',response=r,correct=v)
                for c,r,v in [('baseline','The answer is 42',False),('L31-answer-a1','Final answer: 42',True)]]
            results=report(rows,'gsm8k',bank,root/'out','fixture',Filter)
            self.assertEqual([r['gain_pp'] for r in results],[100.,0.,0.])
            self.assertEqual(len((root/'out/disagreements.jsonl').read_text().splitlines()),1)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Needs CPU PyTorch')
class ExtensionRuntimeTests(unittest.TestCase):
    def test_existing_vector_applied_once_only_to_answers_seed42_and_resumes(self):
        import numpy as np
        import torch
        from evaluate_commonsense_audio import PROMPT
        from extract_audio_vectors import GROUPS
        class Block(torch.nn.Module):
            def forward(self,h):return (h,None)
        blocks=torch.nn.ModuleList([Block() for _ in range(32)])
        model=SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))))
        calls=[];rng=[]
        def generate(audio,sr,model,tokenizer,features,prompt,limit):
            import random
            rng.append((random.random(),float(np.random.random()),float(torch.rand(()))))
            h=torch.zeros(1,520,3)
            for layer in blocks:h=layer(h)[0]
            calls.append((h.clone(),prompt,limit))
            return 'Final answer: B' if h[0,-1,0]>0 else 'Final answer: A'
        original_to=torch.Tensor.to
        def cpu_to(tensor,*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return original_to(tensor,*args,**kwargs)
        modules={'soundfile':SimpleNamespace(read=lambda *a,**k:(np.zeros(20,dtype=np.float32),16000)),
            'extract_vectors':SimpleNamespace(load_model=lambda:(model,None,None),gen=generate)}
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);vector=root/'style-vectors.pt'
            sums=torch.zeros(32,4,3);sums[31,GROUPS.index('answer')]=2
            counts=torch.zeros(32,dtype=torch.long);counts[31]=2
            torch.save(dict(pair_mean_sums={'angry':sums},pair_counts={'angry':counts},prompt=PROMPT,max_new_tokens=512),vector)
            recipe=dict(vector=str(vector),recordings=[dict(question_id='new',audio_file='fake.wav',voice='Jenny',audio_path='neutral.wav',answer='B')])
            with patch.dict('sys.modules',modules),patch.object(torch.Tensor,'to',cpu_to):
                run(recipe,root);run(recipe,root)
            self.assertEqual(len(calls),2)
            self.assertEqual(rng[0],rng[1])
            self.assertTrue(torch.equal(calls[0][0],torch.zeros(1,520,3)))
            self.assertTrue(torch.equal(calls[1][0][:,:8],torch.zeros(1,8,3)))
            self.assertTrue(torch.equal(calls[1][0][:,-512:],torch.ones(1,512,3)))
            self.assertEqual(calls[1][1:],(PROMPT,512))
            rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
            self.assertEqual([r['seed'] for r in rows],[42,42])
            self.assertEqual([r['correct'] for r in rows],[False,True])
            self.assertTrue(all(not b._forward_hooks for b in blocks))


if __name__=='__main__':unittest.main()
