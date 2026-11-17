import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch
import bbh_candidate_adapters as adapters
import evaluate_bbh_candidates as run


class CandidateTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'audio'; source.mkdir()
        example = dict(input='Three books.\nOptions:\n(A) Red\n(B) Blue\n(C) Green', target='(C)')
        raw = json.dumps(dict(examples=[example]*250)).encode()
        (source/'logical_deduction_three_objects-source.json').write_bytes(raw)
        bank = run.existing.select_questions(raw, 250)
        (source/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bank))
        with wave.open(str(source/'long.wav'), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(100)
            f.writeframes(b'\0\0'*4500)
        rows = [dict(q, provider='azure', voice='Jenny', style=s, audio_path='long.wav')
                for q in bank for s in ('neutral', 'angry')]
        (source/'plan.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return source, hashlib.sha256(raw).hexdigest()

    def test_all_candidates_same_250_full_audio_both_prompts_no_model_import(self):
        with tempfile.TemporaryDirectory() as d:
            source, sha = self.fixture(Path(d))
            with patch.object(run.existing, 'SOURCE_SHA256', sha):
                for model in adapters.MODELS:
                    old = run.make_plan(source, model, 'original')
                    new = run.make_plan(source, model, 'explicit')
                    self.assertEqual(old['recordings'], new['recordings'])
                    self.assertEqual(len(old['recordings']), 500)
                    self.assertTrue(all(r['seconds'] == 45 for r in old['recordings']))
                    self.assertEqual(old['interventions'], [])
                    self.assertEqual(old['recipe']['model'], adapters.MODELS[model]['model'])
                    self.assertEqual(old['recipe']['revision'], adapters.MODELS[model]['revision'])
                    self.assertEqual(new['prompt'].replace('Give concise step-by-step reasoning with only the necessary deductions.',
                        "Let's think step by step."), old['prompt'])

    def test_full_waveform_loader_resamples_stereo_without_30_second_crop(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'long.wav'
            with wave.open(str(path), 'wb') as f:
                f.setnchannels(2); f.setsampwidth(2); f.setframerate(24000)
                f.writeframes(b'\x01\x00\x03\x00'*(24000*41))
            audio = adapters.load_audio(path)
            self.assertEqual(len(audio), 16000*41)
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.ndim, 1)

    def test_complete_audio_is_reused_without_tts_or_plan_changes(self):
        with tempfile.TemporaryDirectory() as d:
            source, sha = self.fixture(Path(d)); before = (source/'plan.jsonl').read_bytes()
            with patch.object(run.existing, 'SOURCE_SHA256', sha), \
                 patch.object(run.existing, 'generate_audio') as tts:
                run.audio(source)
            tts.assert_not_called()
            self.assertEqual(before, (source/'plan.jsonl').read_bytes())

    def test_checkpoint_preflight_rejects_insufficient_space_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict('os.environ', {'HF_HUB_CACHE': d}), \
                 patch.object(adapters.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)), \
                 patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaisesRegex(RuntimeError, 'No results or caches have been deleted'):
                    adapters.checkpoint('flamingo')
                download.assert_not_called()

    def test_cached_indexed_checkpoint_does_not_require_download_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            spec = adapters.MODELS['phi4']
            snapshot = Path(d)/('models--'+spec['model'].replace('/', '--'))/'snapshots'/spec['revision']
            snapshot.mkdir(parents=True)
            for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json'):
                (snapshot/name).write_text('{}')
            (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model-01.safetensors', 'y': 'model-02.safetensors'}}))
            (snapshot/'model-01.safetensors').write_bytes(b'complete shard')
            (snapshot/'model-02.safetensors').write_bytes(b'complete shard')
            with patch.dict('os.environ', {'HF_HUB_CACHE': d}), patch('huggingface_hub.snapshot_download') as download:
                with patch.object(adapters.shutil, 'disk_usage', return_value=SimpleNamespace(free=3.7*2**30)):
                    self.assertEqual(adapters.checkpoint('phi4'), str(snapshot))
                    download.assert_not_called()
                    (snapshot/'model-02.safetensors').unlink()
                    with self.assertRaisesRegex(RuntimeError, 'uncached weights'): adapters.checkpoint('phi4')

    def test_native_adapters_use_audio_and_exact_prompt_without_gold(self):
        class Batch(dict):
            def to(self, device, **kwargs): return self
        class Processor:
            def __call__(self, **kwargs):
                self.inputs = kwargs; return Batch(input_ids=torch.tensor([[1, 2]]))
            def batch_decode(self, ids, **kwargs):
                self.decoded = ids; return ['The answer is (C).']
            def apply_chat_template(self, conversation, **kwargs):
                self.conversation = conversation; return 'native audio chat'
        class Model:
            device = 'cpu'
            def eval(self): return self
            def cuda(self): return self
            def generate(self, **kwargs):
                self.kwargs = kwargs; return torch.tensor([[1, 2, 7, 8]])
            def chat(self, **kwargs):
                self.kwargs = kwargs; return 'The answer is (C).'
        for key in adapters.MODELS:
            processor, model = Processor(), Model()
            processor.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [7, 8])
            auto = lambda value: SimpleNamespace(from_pretrained=lambda *args, **kwargs: value)
            stub = SimpleNamespace(AutoProcessor=auto(processor), AutoModelForCausalLM=auto(model),
                AutoModel=auto(model), AudioFlamingo3ForConditionalGeneration=auto(model),
                GenerationConfig=auto(SimpleNamespace()))
            with patch.dict('sys.modules', {'transformers': stub}), \
                 patch.object(adapters, 'checkpoint', return_value='/pinned/local/snapshot'), \
                 patch.object(adapters, 'load_audio', return_value=np.zeros(16000, dtype=np.float32)):
                response, tokens = adapters.make_generator(key)(
                    {'audio_file': '/recording.wav', 'reference_answer': 'DO NOT LEAK'}, 'EXACT TASK PROMPT')
            self.assertEqual((response, tokens), ('The answer is (C).', 2))
            self.assertFalse(model.kwargs['do_sample'])
            self.assertEqual(model.kwargs['max_new_tokens'], 512)
            if key == 'phi4':
                self.assertEqual(processor.inputs['text'], '<|user|><|audio_1|>EXACT TASK PROMPT<|end|><|assistant|>')
                self.assertEqual(processor.inputs['audios'][0][1], 16000)
                self.assertEqual(processor.decoded.tolist(), [[7, 8]])
            elif key == 'minicpm':
                self.assertEqual(model.kwargs['msgs'][0]['content'][0], 'EXACT TASK PROMPT')
                self.assertFalse(model.kwargs['generate_audio'])
                self.assertFalse(model.kwargs['enable_thinking'])
            else:
                self.assertEqual(processor.conversation[0]['content'][0]['text'], 'EXACT TASK PROMPT')
                self.assertEqual(processor.inputs['sampling_rate'], 16000)

    def test_resume_keeps_prompt_model_audio_seed_and_budget_matched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, sha = self.fixture(root)
            with patch.object(run.existing, 'SOURCE_SHA256', sha):
                plan = run.make_plan(source, 'phi4', 'original')
            out = root/'out'; out.mkdir(); task = plan['recordings'][0]
            saved = dict(task, model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                prompt=plan['prompt'], seed=42, max_new_tokens=512, interventions=[], response='The answer is (C).')
            (out/'results.jsonl').write_text(json.dumps(saved)+'\n')
            self.assertEqual(len(run.pending([(out, plan)])), 499)
            saved['seed'] = 43
            (out/'results.jsonl').write_text(json.dumps(saved)+'\n')
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                run.pending([(out, plan)])

    def test_concurrent_budget_checks_headroom_and_caps_allocator(self):
        gib = 2**30
        with patch.object(torch.cuda, 'mem_get_info', side_effect=[(20*gib,45*gib),(30*gib,45*gib)]), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction') as cap, \
             patch.object(run.time, 'sleep') as wait:
            run.concurrent_budget(22)
        wait.assert_called_once_with(30)
        cap.assert_called_once_with(22/45, device=0)

    def test_concurrent_budget_rejects_impossible_cap(self):
        with patch.object(torch.cuda, 'mem_get_info', return_value=(20*2**30,20*2**30)):
            with self.assertRaisesRegex(ValueError, 'smaller than total'):
                run.concurrent_budget(22)


if __name__ == '__main__':
    unittest.main()
