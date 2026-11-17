from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import collect_phi4_reasoning_emotion as run
from phi4_answer_only import DIRECTIVE
from evaluate_gsm_candidates import ORIGINAL


class Phi4DirectionTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';source.mkdir();out=root/'out';out.mkdir()
        records=[dict(question_id=f'q{i:03d}',voice='Jenny',style=s,audio_sha256=f'{i}-{s}',audio_file=f'/audio/{i}-{s}.wav')
                 for i in range(120) for s in ('neutral','angry')]
        plan=dict(recipe=run.MODELS['phi4'],prompt=ORIGINAL,recordings=records,
                  bank={f'q{i:03d}':dict(question='Question',answer='2') for i in range(120)})
        (source/'run_config.json').write_text(json.dumps(plan))
        rows=[dict(r,prompt=ORIGINAL,seed=42,max_new_tokens=512,model=run.MODELS['phi4']['model'],
                   revision=run.MODELS['phi4']['revision'],interventions=[],response='wrong answer') for r in records]
        (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return SimpleNamespace(source_run=source,output=out,task='gsm8k')

    def test_same100_all_three_conditions_no_correctness_filter_frozen_resume(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));plan=run.prepare(args)
            self.assertEqual(plan['questions'],100);self.assertEqual(plan['forwards'],300)
            self.assertEqual(len(plan['pairs']),100)
            self.assertEqual(len({p['question_id'] for p in plan['pairs']}),100)
            self.assertEqual(plan['conditions'],list(run.CONDITIONS))
            self.assertIn(DIRECTIVE,plan['answer_only_prompt'])
            self.assertIn('#### <number>',plan['answer_only_prompt'])
            # Resume does not reread the growing or removed raw source file.
            (args.source_run/'results.jsonl').unlink()
            self.assertEqual(plan,run.prepare(args))

    def test_prefill_hooks_capture_answer_and_audio_without_generation_or_cache(self):
        class Block(torch.nn.Module):
            def forward(self,h):return (h+1,None)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layers=torch.nn.ModuleList([Block(),Block()])
            def forward(self,**inputs):
                self.kwargs=inputs
                h=torch.arange(15,dtype=torch.float32).reshape(1,5,3)
                for block in self.layers:h=block(h)[0]
                return SimpleNamespace(logits=torch.zeros(1,1,8))
        model=Model();inputs=dict(input_ids=torch.tensor([[1,200011,200011,10,11]]),
            audio_embed_sizes=torch.tensor([2]),attention_mask=torch.ones(1,5,dtype=torch.long))
        with patch.object(run,'decoder_layers',return_value=model.layers):
            result=run.capture(model,inputs)
        self.assertFalse(model.kwargs['use_cache'])
        self.assertEqual(model.kwargs['num_logits_to_keep'],1)
        self.assertEqual(result['input_tokens'],5);self.assertEqual(result['audio_tokens'],2)
        torch.testing.assert_close(result['answer'][0],torch.tensor([13.,14.,15.]))
        torch.testing.assert_close(result['audio'][0],torch.tensor([5.5,6.5,7.5]))
        self.assertTrue(all(len(b._forward_hooks)==0 for b in model.layers))

    def test_summary_directions_sign_and_cross_half_diagnostics(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(d));plan=run.prepare(args)
            states=args.output/'states';states.mkdir()
            reason=torch.tensor([1.,0.,0.]).repeat(32,1)
            emotion=2*reason
            for i,pair in enumerate(plan['pairs']):
                nr=torch.ones(32,3)*(i+1)
                for c,h in zip(run.CONDITIONS,(nr,nr-reason,nr+emotion)):
                    torch.save(dict(protocol=run.PROTOCOL,question_id=pair['question_id'],condition=c,
                        answer=h,audio=torch.ones(32,3)),states/f'{i:03d}-{c}.pt')
            run.summarize(args.output)
            vectors=torch.load(args.output/'style-vectors.pt',weights_only=True)
            torch.testing.assert_close(vectors['reasoning_minus_answer_only'],reason)
            torch.testing.assert_close(vectors['angry_minus_neutral'],emotion)
            torch.testing.assert_close(vectors['neutral_minus_angry'],-emotion)
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(summary['pairs'],100)
            self.assertEqual(len(summary['layers']),32)
            for layer in summary['layers']:
                self.assertEqual(layer['cosine_angry_neutral_vs_reasoning'],1.)
                self.assertEqual(layer['cosine_neutral_angry_vs_reasoning'],-1.)
                self.assertEqual(layer['cross_half_reason_even_emotion_odd'],1.)
                self.assertEqual(layer['signed_projection_coefficient'],2.)
                self.assertEqual(layer['audio_instruction_difference_norm'],0.)

    def test_zero_direction_has_no_defined_cosine(self):
        self.assertIsNone(run.cosine(torch.zeros(3),torch.ones(3)))


if __name__=='__main__':unittest.main()
