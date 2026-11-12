"""Check live diffusion timing, temporal averaging, and conservative cache reuse."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from collect_temporal_vectors import paired_bins, collect, fit
from compare_cloud_accuracy import PROMPT
from evaluate_audio_steering import copy_baselines
from run_large_steering import conditions, live_rows, reuse_always
import run_large_steering as queue
from steer import diffusion_progress, make_steering_hook, schedule_active, read_rows

try:
    import torch
except ImportError:
    torch = None


class ReportTests(unittest.TestCase):
    def test_queue_wires_all_stages_in_order_without_all_forward_or_answer_training(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, collection = root/'source', root/'collection'
            source.mkdir()
            (collection/'pairs').mkdir(parents=True)
            audio, vector = root/'clip.wav', root/'vector.pt'
            audio.touch()
            vector.touch()
            records = [dict(clip_id=f'c{i}', question_id=f'test{i}', audio_path=str(audio), answer='18') for i in range(460)]
            (source/'run_config.json').write_text(json.dumps(dict(recordings=records, settings=dict(vectors=str(vector)))))
            pairs = [dict(question_id=f'train{i}', neutral_id=f'n{i}', angry_id=f'a{i}') for i in range(425)]
            manifest = {}
            for i, pair in enumerate(pairs):
                (collection/'pairs'/f'{i:05d}.pt').touch()
                for key in ('neutral_id', 'angry_id'):
                    manifest[pair[key]] = dict(question_id=pair['question_id'], audio_path=str(audio))
            (collection/'collection-config.json').write_text(json.dumps(dict(pairs=pairs, manifest=manifest)))
            with patch.object(queue, 'ROOT', root/'queue'), patch.object(queue, 'SOURCE', source), \
                    patch.object(queue, 'COLLECTION', collection), patch.object(sys, 'argv', ['test']), \
                    patch.object(queue, 'run') as execute, patch.object(queue, 'clock_smoke'), \
                    patch.object(queue, 'reuse_always'), patch.object(queue, 'report'), patch.object(queue.signal, 'signal'):
                queue.main()
            calls = [list(call.args) for call in execute.call_args_list]
            self.assertEqual(calls[0][1], 'collect_temporal_vectors.py')
            self.assertIn('--prepare-only', calls[0])
            self.assertEqual(calls[1][1], 'evaluate_audio_steering.py')
            self.assertEqual(sum(c[1] == 'extract_benefit_vectors.py' for c in calls), 2)
            self.assertEqual(sum(c[1] == 'collect_temporal_vectors.py' for c in calls), 2)
            self.assertTrue(all('train_answer_vectors.py' not in c and 'all' not in c for c in calls))
            self.assertEqual(json.loads((root/'queue/current-stage.json').read_text())['status'], 'finished')

    def test_resume_repairs_only_incomplete_final_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results.jsonl'
            path.write_bytes(b'{"a":1}\n{"incomplete":"\xe2')
            self.assertEqual(read_rows(path, repair=True), [{'a': 1}])
            with path.open('a') as stream:
                stream.write('{"b":2}\n')
            self.assertEqual(read_rows(path), [{'a': 1}, {'b': 2}])
            path.write_text('{"a":1}')
            read_rows(path, repair=True)
            with path.open('a') as stream:
                stream.write('{"b":2}\n')
            self.assertEqual(read_rows(path), [{'a': 1}, {'b': 2}])
            path.write_text('{broken}\n{"a":1}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_rows(path, repair=True)

    def test_baseline_resume_ignores_timing_metadata_but_rejects_different_answer(self):
        record = dict(clip_id='c', question_id='q', answer='18')
        row = dict(clip_id='c', question_id='q', condition='baseline', seed=42, prompt=PROMPT,
                   max_new_tokens=512, decoder='fast_dllm', steered_forwards=0, interventions=[],
                   reference_answer='18', response='Final answer: 18', correct=True, generation_seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results.jsonl'
            path.write_text(json.dumps(row)+'\n')
            copy_baselines(path, {'c': dict(row, generation_seconds=20, mode='phase')}, {'c': record})
            self.assertEqual(read_rows(path), [row])
            with self.assertRaisesRegex(ValueError, 'Conflicting same-seed baseline'):
                copy_baselines(path, {'c': dict(row, response='Final answer: 17', correct=False)}, {'c': record})

    def test_live_report_ignores_only_unfinished_last_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results.jsonl'
            path.write_text('{"a":1}\n{"unfinished":')
            self.assertEqual(live_rows(path), [{'a': 1}])
            path.write_text('{broken}\n{"a":1}\n')
            with self.assertRaises(json.JSONDecodeError):
                live_rows(path)

    def test_schedules_are_independent_conditions_with_three_targets(self):
        rows = conditions(['always', 'early', 'middle', 'late', 'first_forward', 'periodic2'])
        self.assertEqual(len(rows), 19)
        self.assertEqual(len({r['name'] for r in rows}), 19)
        self.assertTrue(all(len(r['interventions']) == 1 for r in rows[1:]))

    def test_reuse_requires_literal42_and_same_checkpoint_and_recording(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            vector = root/'v.pt'
            vector.write_bytes(b'checkpoint')
            stat = vector.stat()
            record = dict(clip_id='c', question_id='q', answer='18', audio_path='same.wav')
            config = dict(recordings=[record], settings=dict(vectors=str(vector), vector_bytes=stat.st_size,
                vector_mtime_ns=stat.st_mtime_ns, seed_policy='constant_before_every_generation', mode='fixed'))
            destination, good, wrong_seed, wrong_vector = [root/n for n in ('dest', 'good', 'seed', 'vector')]
            for path in (destination, good, wrong_seed, wrong_vector):
                path.mkdir()
                (path/'run_config.json').write_text(json.dumps(config))
            row = dict(clip_id='c', condition='L12-acoustic-a1', seed=42, prompt=PROMPT, max_new_tokens=512,
                       decoder='fast_dllm', mode='fixed', interventions=conditions()[2]['interventions'],
                       reference_answer='18', response='Final answer: 18', correct=True)
            (good/'results.jsonl').write_text(json.dumps(row)+'\n')
            (wrong_seed/'results.jsonl').write_text(json.dumps(dict(row, seed=999))+'\n')
            (wrong_vector/'results.jsonl').write_text(json.dumps(row)+'\n')
            changed = json.loads(json.dumps(config))
            changed['settings']['vector_bytes'] += 1
            (wrong_vector/'run_config.json').write_text(json.dumps(changed))
            reuse_always(destination, vector, [wrong_seed, wrong_vector])
            self.assertEqual(live_rows(destination/'results.jsonl'), [])
            reuse_always(destination, vector, [good, good])
            self.assertEqual(live_rows(destination/'results.jsonl'), [row])


@unittest.skipIf(torch is None, 'Needs CPU PyTorch')
class TimingTests(unittest.TestCase):
    def test_collection_exports_both_methods_and_resumes_without_loading_model(self):
        class Block(torch.nn.Module):
            def forward(self, hidden):
                return (hidden, None)

        class LLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(126337, 2)
                self.model = SimpleNamespace(transformer=SimpleNamespace(blocks=[Block() for _ in range(32)]))

            def get_input_embeddings(self):
                return self.emb

            def forward(self, inputs_embeds):
                hidden = inputs_embeds
                for layer in self.model.transformer.blocks:
                    hidden = layer(hidden)[0]
                return hidden

        model = SimpleNamespace(llm_model=LLM(), config=SimpleNamespace(prompt_size=2))
        decoder = ModuleType('src.fast_dllm.diffa_generate')
        original_insert = lambda *args: None
        decoder.insert_audio_embedding = original_insert
        calls = []

        def gen(audio, rate, model, tokenizer, features, prompt, tokens):
            calls.append(audio)
            decoder.insert_audio_embedding(None, torch.zeros(1, 4, 2), torch.tensor([1]))
            for filled in (0, 12, 24):
                ids = torch.full((1, 517), 126336)
                ids[0, -512:-512+filled] = 1
                hidden = model.llm_model.get_input_embeddings()(ids)
                model.llm_model(inputs_embeds=hidden + audio)
            return 'Final answer: 18'

        modules = {'soundfile': SimpleNamespace(read=lambda path, **kw: (1 if path == 'angry' else 0, 16000)),
                   'src': ModuleType('src'), 'src.fast_dllm': ModuleType('src.fast_dllm'),
                   'src.fast_dllm.diffa_generate': decoder,
                   'extract_vectors': SimpleNamespace(gen=gen, load_model=lambda: (model, None, None))}
        pair = dict(question_id='q', neutral_id='n', angry_id='a')
        plan = dict(pairs=[pair], prompt=PROMPT, manifest={
            'n': dict(question_id='q', audio_path='neutral'), 'a': dict(question_id='q', audio_path='angry')})
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), redirect_stdout(io.StringIO()):
            output = Path(directory)
            (output/'pairs').mkdir()
            collect(plan, output, torch)
            collect(plan, output, torch)
            self.assertEqual(calls, [0, 1])
            fit(plan, output, torch)
            state = torch.load(output/'phase/style-vectors.pt', weights_only=True)
            self.assertEqual(state['temporal_counts']['angry'][11].tolist(), [1, 1, 1])
            self.assertTrue(torch.allclose(state['temporal_sums']['angry'][11, :, 3], torch.ones(3, 2)))
            windows = torch.load(output/'forward_windows/style-vectors.pt', weights_only=True)
            self.assertEqual(windows['temporal_counts']['angry'][12].tolist(), [1])
            self.assertTrue(torch.allclose(windows['pair_mean_sums']['angry'][12, 0], torch.ones(2)))
            self.assertIs(decoder.insert_audio_embedding, original_insert)
            self.assertTrue(all(not layer._forward_hooks for layer in model.llm_model.model.transformer.blocks))
            single=output/'layer31';(single/'pairs').mkdir(parents=True)
            single_plan=dict(plan,targets=[(31,'answer')])
            collect(single_plan,single,torch)
            fit(single_plan,single,torch,modes=['forward_windows'])
            fitted=torch.load(single/'forward_windows/style-vectors.pt',weights_only=True)
            self.assertEqual(fitted['collected_targets'],[(31,'answer')])
            self.assertEqual(fitted['pair_counts']['angry'][31].item(),1)
            self.assertEqual(fitted['pair_counts']['angry'][11].item(),0)
            self.assertTrue(torch.allclose(fitted['temporal_sums']['angry'][31,0,3],torch.ones(2)))
            self.assertFalse((single/'phase').exists())

    def test_clock_observes_active_block_and_resets_periodic_counter(self):
        class LLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(100, 2)

            def get_input_embeddings(self):
                return self.emb

            def forward(self, inputs_embeds):
                return inputs_embeds.sum()

        model = type('Model', (), {})()
        model.llm_model = LLM()
        sequences = [[99]*8, [1, 99, 99, 99, 99, 99, 99, 99],
                     [1, 1, 99, 99, 99, 99, 99, 99], [1, 1, 1, 99, 99, 99, 99, 99],
                     [1, 1, 1, 1, 99, 99, 99, 99]]
        outputs = [model.llm_model(model.llm_model.emb(torch.tensor([ids]))).item() for ids in sequences]
        observed, periodic, actual = [], [], []
        with diffusion_progress(model, answer_tokens=8, block_length=4, mask_id=99) as progress:
            for ids in sequences:
                actual.append(model.llm_model(model.llm_model.emb(torch.tensor([ids]))).item())
                observed.append((progress['forward'], progress['block'], progress['block_forward'], progress['phase']))
                periodic.append(schedule_active('periodic2', progress))
        self.assertEqual(observed, [(1,0,1,0),(2,0,2,0),(3,0,3,1),(4,0,4,2),(5,1,1,0)])
        self.assertEqual(periodic, [True, False, True, False, True])
        self.assertEqual(actual, outputs)
        self.assertFalse(model.llm_model._forward_pre_hooks)
        self.assertFalse(model.llm_model.emb._forward_pre_hooks)

    def test_temporal_direction_and_gates_use_current_forward(self):
        bins = torch.tensor([[1., 2.], [3., 4.], [9., 10.]])
        output = (torch.zeros(1, 4, 2), None)
        progress = dict(forward=21, block_forward=2, phase=1)
        hook, state = make_steering_hook(bins, None, 'forward_windows', 'answer', 1, {}, 2, progress)
        changed = hook(None, None, output)
        self.assertTrue(torch.equal(changed[0][0, -2:], bins[1].repeat(2, 1)))
        self.assertTrue(torch.equal(output[0], torch.zeros_like(output[0])))
        progress['forward'] = 200
        self.assertTrue(torch.equal(hook(None, None, output)[0][0, -1], bins[-1]))
        gated, state = make_steering_hook(bins, None, 'phase', 'answer', 1, {}, 2, progress, 'early')
        self.assertIsNone(gated(None, None, output))
        progress['phase'] = 0
        self.assertTrue(torch.equal(gated(None, None, output)[0][0, -1], bins[0]))
        self.assertEqual(state['steered'], 1)

    def test_pair_bins_drop_longer_tail_and_phase_match_only_shared_prefix(self):
        neutral = torch.arange(4.).reshape(4, 1, 1)
        angry = torch.tensor([10., 11., 12., 13., 999.]).reshape(5, 1, 1)
        result = paired_bins(neutral, angry, [0,0,1,2], [0,1,1,2,2], torch, window_size=2)
        self.assertEqual(result['matched_forwards'], 4)
        self.assertTrue(torch.equal(result['forward_windows'].flatten(), torch.tensor([10.,10.])))
        self.assertTrue(torch.equal(result['phase'].flatten(), torch.tensor([10.,9.,10.])))
        self.assertEqual(result['phase_valid'], [True, True, True])


if __name__ == '__main__':
    unittest.main()
