import io
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

    def forward(self, **kwargs):
        self.seen.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
        hidden = kwargs['text_input_ids'].float().unsqueeze(-1).repeat(1, 1, 3)
        hidden += kwargs['position_ids'].float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return None, None, ('KV', len(self.seen))


def inputs(length=4):
    return dict(input_ids=torch.arange(length).unsqueeze(0), text_input_ids=torch.zeros(1, length, dtype=torch.long),
                position_ids=torch.arange(length).unsqueeze(0), whisper_input_feature=[torch.ones(1, 2, 3)],
                is_continuous_mask=torch.zeros(1, length, dtype=torch.bool))


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
                torch.save(dict(emotion=torch.full((28, 3), val), matched_emotion=torch.full((28, 3), val)),
                           run.cache_file(output, pair))
            _, vectors = run.mean_vectors(dict(pairs=pairs), output)
            torch.testing.assert_close(vectors['emotion'], torch.full((28, 3), 4.))

    def test_base_comparison_reuses_exact_encoded_neutral_inputs_and_history(self):
        model = SimpleNamespace(alm=FakeALM(offset=2))
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); (output/'pairs').mkdir()
            pair = dict(question_id='a', voice='v')
            neutral = dict(inputs=inputs(), calls=[(3, 0), (999, 91)], means=torch.ones(2, 3))
            torch.save(dict(neutral=neutral), run.cache_file(output, pair))
            with patch.object(run, 'decoder_layers', side_effect=lambda m: m.alm.model.layers):
                run.collect_base(dict(pairs=[pair]), output, model)
            result = torch.load(run.cache_file(output, pair, base=True), weights_only=True)
            torch.testing.assert_close(result['checkpoint_difference'], neutral['means']-result['base_means'])
            torch.testing.assert_close(model.alm.seen[0]['input_ids'], neutral['inputs']['input_ids'])
            torch.testing.assert_close(model.alm.seen[0]['whisper_input_feature'][0], neutral['inputs']['whisper_input_feature'][0])
            self.assertEqual(int(model.alm.seen[1]['text_input_ids'][0, 0]), 91)

    def test_collection_stops_on_changed_native_response_before_saving_vectors(self):
        pair = dict(question_id='q', voice='v', neutral={'response': 'old neutral'}, angry={'response': 'old angry'})
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'trace_generation',
                side_effect=[dict(response='DIFFERENT'), dict(response='old angry')]):
            output = Path(d); (output/'pairs').mkdir()
            with self.assertRaisesRegex(ValueError, 'Native rerun differs'):
                run.collect(dict(pairs=[pair], prompt='prompt'), output, object())
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
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'LAYERS', (0, 1)), \
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


if __name__ == '__main__':
    unittest.main()
