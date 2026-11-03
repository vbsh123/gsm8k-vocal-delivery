"""Check simultaneous interventions, stage order and requested follow-up conditions."""

import unittest
from types import SimpleNamespace

from run_steering_followup import build_stages, install_interventions
from test_steering import Hidden, Vector


class Layer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def run(self, hidden):
        output = (hidden, None)
        for hook in list(self.hooks):
            changed = hook(self, (), output)
            if changed is not None:
                output = changed
        return output[0]


class FollowupTests(unittest.TestCase):
    bounds = dict(start=1, acoustic_end=3, end=5)

    def test_default_plan_is_exact_six_tests_then_26_semantic_tests(self):
        stages = build_stages(list(range(26)))
        self.assertEqual(len(stages[0]["conditions"]), 7)
        self.assertEqual(len(stages[1]["conditions"]), 26)
        self.assertEqual([c["name"] for c in stages[0]["conditions"]], [
            "baseline", "L4-acoustic-a0.5", "L2-answer-a0.5", "L18-answer-a0.5", "L20-answer-a0.5",
            "L4-acoustic+L18-answer-a1", "L18-answer+L20-answer-a0.5"])
        all_conditions = [c for stage in stages for c in stage["conditions"]]
        self.assertEqual(len({c["name"] for c in all_conditions}), 33)
        self.assertEqual(200 * len(all_conditions), 6600)
        for layer, condition in enumerate(stages[1]["conditions"]):
            self.assertEqual(condition["interventions"], [dict(layer=layer, group="semantic", strength=1)])

    def test_higher_strength_plan_has_only_the_requested_five_interventions(self):
        stages = build_stages(list(range(26)), "higher-strength")
        self.assertEqual(len(stages), 1)
        conditions = stages[0]["conditions"]
        self.assertEqual([c["name"] for c in conditions], [
            "baseline", "L4-acoustic-a2", "L18-answer-a2", "L20-answer-a2",
            "L4-acoustic+L18-answer-a2", "L18-answer+L20-answer-a1"])
        self.assertEqual(conditions[4]["interventions"], [
            dict(layer=4, group="acoustic", strength=2), dict(layer=18, group="answer", strength=2)])
        self.assertEqual(conditions[5]["interventions"], [
            dict(layer=18, group="answer", strength=1), dict(layer=20, group="answer", strength=1)])
        self.assertEqual(200 * len(conditions), 1200)

    def test_acoustic_and_answer_are_steered_in_same_forward_and_cleaned_up(self):
        layers = [Layer() for _ in range(21)]
        original = Hidden([[0, 0] for _ in range(7)])
        directions = {(4, "acoustic"): Vector([2, 2]), (18, "answer"): Vector([4, 4])}
        targets = [dict(layer=4, group="acoustic", strength=1), dict(layer=18, group="answer", strength=1)]
        with install_interventions(layers, directions, targets, self.bounds, 2) as states:
            hidden = original
            for layer in layers:
                hidden = layer.run(hidden)
            self.assertEqual(hidden.rows[1:3], [[2, 2], [2, 2]])
            self.assertEqual(hidden.rows[-2:], [[4, 4], [4, 4]])
            self.assertEqual(hidden.rows[3:5], [[0, 0], [0, 0]])
            self.assertEqual([s["state"]["steered"] for s in states], [1, 1])
        self.assertTrue(all(not layer.hooks for layer in layers))
        self.assertTrue(all(row == [0, 0] for row in original.rows))

    def test_two_answer_layers_use_their_own_vectors_at_half_strength(self):
        layers = [Layer() for _ in range(21)]
        directions = {(18, "answer"): Vector([2, 2]), (20, "answer"): Vector([4, 4])}
        targets = [dict(layer=18, group="answer", strength=0.5), dict(layer=20, group="answer", strength=0.5)]
        with install_interventions(layers, directions, targets, self.bounds, 2):
            hidden = Hidden([[0, 0] for _ in range(7)])
            for layer in layers:
                hidden = layer.run(hidden)
            self.assertEqual(hidden.rows[-2:], [[3, 3], [3, 3]])
            self.assertEqual(hidden.rows[:5], [[0, 0] for _ in range(5)])

    def test_baseline_counts_forwards_without_modification(self):
        layers = [Layer()]
        hidden = Hidden([[0, 0] for _ in range(7)])
        with install_interventions(layers, {}, [], {}, 2) as states:
            self.assertIs(layers[0].run(hidden), hidden)
            self.assertEqual(states[0]["state"], dict(forwards=1, steered=0))
        self.assertEqual(layers[0].hooks, [])

    def test_all_hooks_removed_if_generation_or_registration_fails(self):
        for registration_failure in [False, True]:
            layers = [Layer(), None if registration_failure else Layer()]
            targets = [dict(layer=i, group="answer", strength=1) for i in [0, 1]]
            directions = {(i, "answer"): Vector([1, 1]) for i in [0, 1]}
            with self.assertRaises((ValueError, AttributeError)):
                with install_interventions(layers, directions, targets, {}, 2):
                    raise ValueError("generation failed")
            self.assertEqual(layers[0].hooks, [])
            if layers[1] is not None:
                self.assertEqual(layers[1].hooks, [])

    def test_invalid_semantic_layer_lists_are_rejected(self):
        for layers in [[-1], [4, 4]]:
            with self.assertRaises(ValueError):
                build_stages(layers)
