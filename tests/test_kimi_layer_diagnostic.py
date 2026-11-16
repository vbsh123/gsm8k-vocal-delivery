import io
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import kimi_bbh_layer_diagnostic as run


class Block(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, hidden):
        return hidden+self.amount, 'cache-kept'


class FakeALM(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([Block(1+offset), Block(2)]))
        self.seen = []
        self.hidden_history = []

    def forward(self, **kwargs):
        self.seen.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
        hidden = kwargs['text_input_ids'].float().unsqueeze(-1).repeat(1, 1, 3)
        hidden += kwargs['position_ids'].float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        self.hidden_history.append(hidden.detach().clone())
        return None, None, ('KV', len(self.seen))


def inputs(length=4):
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 1:3] = True
    return dict(input_ids=torch.arange(length).unsqueeze(0), text_input_ids=torch.zeros(1, length, dtype=torch.long),
                position_ids=torch.arange(length).unsqueeze(0), whisper_input_feature=[torch.ones(1, 2, 3)],
                is_continuous_mask=mask)


class KimiLayerTests(unittest.TestCase):
    def test_selects_all_recoveries_not_six_net_wins_or_strict_only(self):
        rows = []
        # 30 recoveries and 24 losses => six net wins, 30 selected pairs.
        for i in range(54):
            for style in ('neutral', 'angry'):
                value = style == ('angry' if i < 30 else 'neutral')
                rows.append(dict(question_id=f'q{i:03d}', voice='Jenny', style=style,
                    harness={'flexible-extract': {'correct': value}, 'strict-match': {'correct': not value}}))
        pairs = run.recovery_pairs(rows)
        self.assertEqual(len(pairs), 30)
        self.assertEqual([p['question_id'] for p in pairs], [f'q{i:03d}' for i in range(30)])

    def test_native_forward_trace_and_teacher_replay_match_without_retokenizing(self):
        model = SimpleNamespace(alm=FakeALM())
        initial = inputs()
        def generate(model, row, prompt):
            model.alm.forward(**initial, past_key_values=None, return_dict=False)
            for i, token in enumerate((91, 72)):
                model.alm.forward(input_ids=torch.tensor([[999]]), text_input_ids=torch.tensor([[token]]),
                    position_ids=torch.tensor([[4+i]]), whisper_input_feature=None, is_continuous_mask=None,
                    past_key_values=('native', i), return_dict=False)
            return 'Unretokenized answer'
        with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers), \
             patch.object(run, 'generate', side_effect=generate):
            trace = run.trace_generation(model, {}, 'prompt')
            result = run.replay_means(model, trace['inputs'], trace['calls'])
        torch.testing.assert_close(trace['means'], result, rtol=0, atol=0)
        self.assertEqual(trace['calls'], [(3, 0), (999, 91), (999, 72)])
        self.assertEqual(model.alm.seen[-2]['text_input_ids'].tolist(), [[91]])
        self.assertEqual(model.alm.seen[-1]['text_input_ids'].tolist(), [[72]])
        self.assertEqual(len(model.alm.model.layers[0]._forward_hooks), 0)

    def test_matched_angry_uses_neutral_tokens_but_own_audio_prefix_positions(self):
        model = SimpleNamespace(alm=FakeALM())
        calls = [(3, 0), (999, 91), (999, 72)]
        with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers):
            run.replay_means(model, inputs(length=8), calls)
        self.assertEqual([int(x['position_ids'][0, -1]) for x in model.alm.seen], [7, 8, 9])
        self.assertEqual([int(x['text_input_ids'][0, -1]) for x in model.alm.seen], [0, 91, 72])
        self.assertIsNone(model.alm.seen[0]['past_key_values'])
        self.assertEqual(model.alm.seen[1]['past_key_values'], ('KV', 1))
        self.assertIsNone(model.alm.seen[1]['whisper_input_feature'])

    def test_hook_only_changes_last_position_keeps_cache_and_always_removes_it(self):
        layer = Block(1)
        original = torch.zeros(1, 4, 3)
        with run.steer_layer(layer, torch.tensor([2., 3., 4.])):
            hidden, cache = layer(original)
        torch.testing.assert_close(hidden[0, :-1], torch.ones(3, 3))
        torch.testing.assert_close(hidden[0, -1], torch.tensor([3., 4., 5.]))
        self.assertEqual(cache, 'cache-kept')
        self.assertEqual(original.sum().item(), 0)
        with self.assertRaises(RuntimeError):
            with run.steer_layer(layer, torch.ones(3)):
                raise RuntimeError('stop')
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_means_last_prediction_position_not_audio_and_equal_question_weight(self):
        layers = [Block(0)]
        with run.capture_means(layers) as (sums, counts):
            layers[0](torch.tensor([[[900., 900.], [2., 4.]]]))
            layers[0](torch.tensor([[[6., 8.]]]))
        torch.testing.assert_close(run.finish_means(sums, counts), torch.tensor([[4., 6.]]))
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); (output/'pairs').mkdir()
            pairs = [dict(question_id='a', voice='v'), dict(question_id='b', voice='v')]
            for pair, val in zip(pairs, (2., 6.)):
                torch.save(dict(emotion=torch.full((28, 3), val), matched_emotion=torch.full((28, 3), val), audio_emotion=torch.full((28, 3), val)),
                           run.cache_file(output, pair))
            _, vectors = run.mean_vectors(dict(collection_pairs=pairs), output)
            torch.testing.assert_close(vectors['emotion'], torch.full((28, 3), 4.))

    def test_base_comparison_reuses_exact_encoded_neutral_inputs_and_history(self):
        model = SimpleNamespace(alm=FakeALM(offset=2))
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); (output/'pairs').mkdir()
            pair = dict(question_id='a', voice='v')
            neutral = dict(inputs=inputs(), calls=[(3, 0), (999, 91)], means=torch.ones(2, 3), audio_means=torch.ones(2, 3))
            torch.save(dict(neutral=neutral), run.cache_file(output, pair))
            with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers):
                run.collect_base(dict(collection_pairs=[pair]), output, model)
            result = torch.load(run.cache_file(output, pair, base=True), weights_only=True)
            torch.testing.assert_close(result['checkpoint_difference'], neutral['means']-result['base_means'])
            torch.testing.assert_close(result['audio_checkpoint_difference'], neutral['audio_means']-result['base_audio_means'])
            torch.testing.assert_close(model.alm.seen[0]['input_ids'], neutral['inputs']['input_ids'])
            torch.testing.assert_close(model.alm.seen[0]['whisper_input_feature'][0], neutral['inputs']['whisper_input_feature'][0])
            self.assertEqual(int(model.alm.seen[1]['text_input_ids'][0, 0]), 91)

    def test_collection_stops_on_changed_native_response_before_saving_vectors(self):
        pair = dict(question_id='q', voice='v', neutral={'response': 'old neutral'}, angry={'response': 'old angry'})
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'trace_generation',
                side_effect=[dict(response='DIFFERENT'), dict(response='old angry')]):
            output = Path(d); (output/'pairs').mkdir()
            with self.assertRaisesRegex(ValueError, 'Native rerun differs'):
                run.collect(dict(collection_pairs=[pair], prompt='prompt'), output, object())
            self.assertFalse(run.cache_file(output, pair).exists())
            mismatch = json.loads((output/'reproduction-mismatch.json').read_text())
            self.assertEqual((mismatch['style'], mismatch['actual']), ('neutral', 'DIFFERENT'))

    def test_sweep_resumes_per_question_layer_and_uses_only_emotion_vector(self):
        pair = dict(question_id='q', voice='v', neutral=dict(question_id='q', voice='v', style='neutral',
            condition='neutral', response='old neutral', harness={'old': 'score'}),
            angry=dict(question_id='q', voice='v', style='angry', condition='angry', response='old angry'))
        plan = dict(pairs=[pair], prompt='prompt', benchmark='original')
        model = SimpleNamespace(alm=FakeALM())
        vectors = dict(emotion=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
                       checkpoint_difference=torch.full((2, 3), 100.))
        seen = []
        def generate(model, row, prompt):
            hidden = torch.zeros(1, 2, 3)
            for layer in model.alm.model.layers:
                hidden = layer(hidden)[0]
            seen.append(hidden[0, -1].tolist())
            return 'new response'
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'LAYERS', (0, 1)), patch.object(run, 'GROUPS', ('answer',)), \
             patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers), \
             patch.object(run, 'generate', side_effect=generate):
            output = Path(d)
            run.sweep(plan, output, model, vectors, max_generations=1)
            run.sweep(plan, output, model, vectors)
            run.sweep(plan, output, model, vectors)
            rows = run.read_rows(output/'results.jsonl')
        self.assertEqual(len(rows), 4)
        self.assertEqual(seen, [[4., 5., 6.], [7., 8., 9.]])
        self.assertEqual([r['response'] for r in rows[:2]], ['old neutral', 'old angry'])
        self.assertTrue(all('harness' not in r for r in rows[2:]))
        self.assertEqual([r['interventions'][0]['layer'] for r in rows[2:]], [0, 1])

    def test_collection_is_fixed_125_and_excludes_every_recovery_test_question(self):
        rows = []
        for i in range(250):
            for style in ('neutral', 'angry'):
                rows.append(dict(question_id=f'q{i:03d}', voice='Jenny', style=style,
                    harness={'flexible-extract': {'correct': style == 'angry' and i < 30}}))
        test = run.recovery_pairs(rows)
        collect = run.collection_pairs(rows, test)
        self.assertEqual(len(test), 30)
        self.assertEqual(len(collect), 125)
        self.assertFalse({p['question_id'] for p in test} & {p['question_id'] for p in collect})
        self.assertEqual(collect, run.collection_pairs(list(reversed(rows)), test))

    def test_audio_collection_uses_only_masked_prefill_slots(self):
        layer = Block(0)
        state = {'mask': torch.tensor([[False, True, True, False]])}
        with run.capture_audio_means([layer], state) as values:
            layer(torch.tensor([[[900., 900.], [2., 4.], [6., 8.], [1000., 1000.]]]))
            state['mask'] = None
            layer(torch.tensor([[[2000., 2000.]]]))
        torch.testing.assert_close(run.finish_audio(values, 1), torch.tensor([[4., 6.]]))
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_audio_steering_preserves_instructions_answer_slot_and_cached_steps(self):
        layer = Block(1)
        state = {'mask': torch.tensor([[False, True, True, False]])}
        with run.steer_layer(layer, torch.tensor([2., 3., 4.]), group='audio', audio_state=state):
            hidden, cache = layer(torch.zeros(1, 4, 3))
            state['mask'] = None
            cached, _ = layer(torch.zeros(1, 1, 3))
        torch.testing.assert_close(hidden[0, 0], torch.ones(3))
        torch.testing.assert_close(hidden[0, -1], torch.ones(3))
        torch.testing.assert_close(hidden[0, 1:3], torch.tensor([[3., 4., 5.], [3., 4., 5.]]))
        torch.testing.assert_close(cached, torch.ones(1, 1, 3))
        self.assertEqual(cache, 'cache-kept')

    def test_native_trace_and_replay_preserve_audio_means_separately_from_answer(self):
        model = SimpleNamespace(alm=FakeALM())
        initial = inputs()
        def generate(model, row, prompt):
            model.alm.forward(**initial, past_key_values=None, return_dict=False)
            model.alm.forward(input_ids=torch.tensor([[999]]), text_input_ids=torch.tensor([[90]]),
                position_ids=torch.tensor([[4]]), whisper_input_feature=None, is_continuous_mask=None,
                past_key_values=('KV', 1), return_dict=False)
            return 'answer'
        with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers), \
             patch.object(run, 'generate', side_effect=generate):
            trace = run.trace_generation(model, {}, 'prompt')
            replay = run.replay_means(model, trace['inputs'], trace['calls'], include_audio=True)
        torch.testing.assert_close(trace['audio_means'], replay['audio_means'])
        torch.testing.assert_close(trace['means'], replay['means'])
        self.assertFalse(torch.equal(trace['audio_means'], trace['means']))

    def test_audio_and_answer_sweeps_use_distinct_vectors_and_resume_independently(self):
        pair = dict(question_id='q', voice='v', neutral=dict(question_id='q', voice='v', style='neutral',
            response='old neutral'), angry=dict(question_id='q', voice='v', style='angry', response='old angry'))
        plan = dict(pairs=[pair], prompt='prompt', benchmark='original')
        vectors = dict(audio_emotion=torch.full((1, 3), 11.), emotion=torch.full((1, 3), 22.))
        calls = []
        def steered(model, row, prompt, layer, group, direction):
            calls.append((layer, group, direction[0].item()))
            return group+' response'
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'LAYERS', (0,)), \
             patch.object(run, 'generate_steered', side_effect=steered):
            output = Path(d)
            run.sweep(plan, output, object(), vectors, max_generations=1)
            run.sweep(plan, output, object(), vectors)
            rows = run.read_rows(output/'results.jsonl')
        self.assertEqual(calls, [(0, 'audio', 11.), (0, 'answer', 22.)])
        self.assertEqual([r['condition'] for r in rows[2:]], ['neutral-Kimi-L0-audio-a1', 'neutral-Kimi-L0-answer-a1'])

    def test_generate_wrapper_supplies_native_audio_mask_and_clears_it_for_cache(self):
        model = SimpleNamespace(alm=FakeALM())
        def generate(model, row, prompt):
            model.alm.forward(**inputs(), past_key_values=None, return_dict=False)
            model.alm.forward(input_ids=torch.tensor([[999]]), text_input_ids=torch.tensor([[0]]),
                position_ids=torch.tensor([[4]]), whisper_input_feature=None, is_continuous_mask=None,
                past_key_values=('KV', 1), return_dict=False)
            return 'answer'
        with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers), \
             patch.object(run, 'generate', side_effect=generate):
            run.generate_steered(model, {}, 'prompt', 0, 'audio', torch.full((3,), 10.))
        prefill, cached = model.alm.hidden_history
        torch.testing.assert_close(prefill[0, 0], torch.full((3,), 3.))
        torch.testing.assert_close(prefill[0, 1], torch.full((3,), 14.))
        torch.testing.assert_close(prefill[0, 2], torch.full((3,), 15.))
        torch.testing.assert_close(prefill[0, 3], torch.full((3,), 6.))
        torch.testing.assert_close(cached[0, 0], torch.full((3,), 7.))

    def test_cosine_report_keeps_audio_and_answer_checkpoint_differences_separate(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'LAYERS', (0, 1)):
            output = Path(d); (output/'pairs').mkdir()
            pair = dict(question_id='collection', voice='v')
            shape = (2, 3)
            data = [dict(neutral=dict(means=torch.full(shape, 2.), audio_means=torch.full(shape, 9.)))]
            torch.save(dict(checkpoint_difference=torch.ones(shape), audio_checkpoint_difference=torch.full(shape, 6.),
                base_means=torch.ones(shape), base_audio_means=torch.full(shape, 3.)), run.cache_file(output, pair, True))
            vectors = dict(emotion=torch.ones(shape), matched_emotion=torch.ones(shape), audio_emotion=torch.full(shape, 3.))
            run.cosine_report(dict(collection_pairs=[pair]), output, data, vectors)
            saved = torch.load(output/'checkpoint-differences.pt', weights_only=True)
            with (output/'layer-cosines.csv').open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['group'] for r in rows}, {'audio', 'answer'})
        torch.testing.assert_close(saved['audio']['checkpoint_difference'], torch.full(shape, 6.))
        torch.testing.assert_close(saved['answer']['checkpoint_difference'], torch.ones(shape))

    def test_emotion_and_checkpoint_collection_never_read_sweep_question_cache(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); (output/'pairs').mkdir()
            collect = dict(question_id='collection', voice='v')
            test = dict(question_id='test', voice='v')
            plan = dict(collection_pairs=[collect], pairs=[test])
            torch.save(dict(emotion=torch.ones(28, 3), matched_emotion=torch.ones(28, 3), audio_emotion=torch.ones(28, 3)), run.cache_file(output, collect))
            torch.save(dict(emotion=torch.full((28, 3), 999.), matched_emotion=torch.full((28, 3), 999.), audio_emotion=torch.full((28, 3), 999.)), run.cache_file(output, test))
            _, vectors = run.mean_vectors(plan, output)
            torch.testing.assert_close(vectors['emotion'], torch.ones(28, 3))
            saved = torch.load(output/'style-vectors.pt', weights_only=True)
            self.assertEqual(saved['question_ids'], ['collection'])


if __name__ == '__main__':
    unittest.main()
