from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import collect_diffa_reasoning_emotion as run
from run_reasoning_transfer import HASH_PROMPT


class DiffaDirectionTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';source.mkdir();output=root/'out';output.mkdir()
        audio=root/'audio.wav';audio.write_bytes(b'original test recording')
        records=[dict(question_id=f'q{i:03d}',voice='Jenny',style='angry' if c=='angry' else 'neutral',
            condition=c,audio_file=str(audio),audio_sha256=run.digest(audio),answer='2')
            for i in range(110) for c in ('neutral','angry','neutral-L31-answer-a1')]
        plan=dict(seed=42,max_new_tokens=512,steps=128,block_length=32,decoder='fast_dllm',prompt=HASH_PROMPT,
                  recordings=records,bank={f'q{i:03d}':dict(question='Question',answer='2') for i in range(110)})
        (source/'run_config.json').write_text(json.dumps(plan))
        return SimpleNamespace(source_run=source,output=output)

    def test_same100_three_conditions_no_steered_or_outcome_filter(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));before=(args.source_run/'run_config.json').read_bytes()
            plan=run.prepare(args)
            self.assertEqual(len(plan['pairs']),100)
            self.assertEqual(plan['forwards'],300)
            self.assertEqual(plan['model'],'zhoujiaming777/DIFFA-2')
            self.assertEqual(plan['groups'],['acoustic','semantic','combined','answer'])
            self.assertEqual(plan['conditions'],list(run.CONDITIONS))
            self.assertIn('#### <number>',plan['answer_only_prompt'])
            self.assertNotIn('step-by-step reasoning',plan['answer_only_prompt'])
            self.assertEqual(before,(args.source_run/'run_config.json').read_bytes())
            (args.source_run/'run_config.json').unlink()
            self.assertEqual(plan,run.prepare(args))

    def model(self):
        class Block(torch.nn.Module):
            def forward(self,h):return (h+1,None)
        class LLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=SimpleNamespace(transformer=SimpleNamespace(blocks=torch.nn.ModuleList([Block(),Block()])))
            def forward(self,inputs_embeds):
                h=inputs_embeds
                for block in self.model.transformer.blocks:h=block(h)[0]
                return SimpleNamespace(logits=h)
        return SimpleNamespace(llm_model=LLM(),config=SimpleNamespace(prompt_size=2))

    def test_first_native_forward_stops_before_commit_captures_groups_restores_hooks(self):
        model=self.model();decoder=SimpleNamespace(insert_audio_embedding=lambda text,speech,lengths:text)
        original=decoder.insert_audio_embedding;committed=[]
        h=torch.arange(520*3,dtype=torch.float32).reshape(1,520,3)
        def generate():
            decoder.insert_audio_embedding(h,torch.zeros(1,4,3),torch.tensor([2]))
            model.llm_model(inputs_embeds=h)
            committed.append(True)
        trace=run.capture_initial(model,generate,decoder)
        self.assertFalse(committed)
        self.assertEqual(trace['answer_tokens'],512)
        self.assertEqual(trace['speech_bounds'],dict(start=2,acoustic_end=4,end=6))
        self.assertEqual(trace['states'].shape,(2,4,3))
        torch.testing.assert_close(trace['states'][0,0],(h[0,2:4]+1).mean(0))
        torch.testing.assert_close(trace['states'][0,3],(h[0,-512:]+1).mean(0))
        self.assertIs(decoder.insert_audio_embedding,original)
        self.assertEqual(len(model.llm_model._forward_hooks),0)
        self.assertEqual(len(model.llm_model._forward_pre_hooks),0)
        self.assertTrue(all(not layer._forward_hooks for layer in model.llm_model.model.transformer.blocks))

    def test_failure_restores_decoder_and_hooks(self):
        model=self.model();decoder=SimpleNamespace(insert_audio_embedding=lambda *a:None)
        original=decoder.insert_audio_embedding
        with self.assertRaisesRegex(ValueError,'inputs_embeds-only'):
            run.capture_initial(model,lambda:model.llm_model(torch.zeros(1,520,3)),decoder)
        self.assertIs(decoder.insert_audio_embedding,original)
        self.assertFalse(model.llm_model._forward_pre_hooks)

    def test_all_group_cosines_and_reverse_direction_are_saved(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            args=self.fixture(Path(d));plan=run.prepare(args);folder=args.output/'states';folder.mkdir()
            r=torch.ones(32,4,3);e=-.5*r
            for i,pair in enumerate(plan['pairs']):
                baseline=torch.ones(32,4,3)*i
                for c,h in zip(run.CONDITIONS,(baseline,baseline-r,baseline+e)):
                    torch.save(dict(protocol=run.PROTOCOL,question_id=pair['question_id'],condition=c,
                        states=h,answer_input_sha256='same initial answer masks'),folder/f'{i:03d}-{c}.pt')
            run.summarize(args.output)
            vectors=torch.load(args.output/'style-vectors.pt',weights_only=True)
            torch.testing.assert_close(vectors['reasoning_minus_answer_only'],r)
            torch.testing.assert_close(vectors['angry_minus_neutral'],e)
            torch.testing.assert_close(vectors['neutral_minus_angry'],-e)
            result=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(result['pairs'],100);self.assertEqual(len(result['layers']),128)
            for row in result['layers']:
                self.assertAlmostEqual(row['cosine_angry_neutral_vs_reasoning'],-1.,places=6)
                self.assertAlmostEqual(row['cosine_neutral_angry_vs_reasoning'],1.,places=6)


if __name__=='__main__':unittest.main()
