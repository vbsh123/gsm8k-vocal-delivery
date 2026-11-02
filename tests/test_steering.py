"""Test held-out selection, hook behavior and paired reporting without a GPU."""

from contextlib import redirect_stdout
import io
import unittest

from steer import make_conditions, make_steering_hook, reuse_questions, select_neutrals, summarize
from prepare_steering_exclusions import extraction_questions


class Vector(list):
    def __rmul__(self, scale):
        return [scale * x for x in self]


class Selection:
    def __init__(self, hidden, positions):
        self.hidden = hidden
        self.positions = positions

    def __iadd__(self, delta):
        for i in range(*self.positions.indices(len(self.hidden.rows))):
            self.hidden.rows[i] = [x + y for x, y in zip(self.hidden.rows[i], delta)]
        return self


class Hidden:
    def __init__(self, rows):
        self.rows = rows

    def clone(self):
        return Hidden([r[:] for r in self.rows])

    def __getitem__(self, key):
        return Selection(self, key[1])

    def __setitem__(self, key, value):
        assert isinstance(value, Selection) and value.hidden is self


class SteeringTests(unittest.TestCase):
    bounds = {"start": 1, "acoustic_end": 3, "end": 5}

    def test_fixed_changes_only_chosen_group_and_preserves_original_and_cache(self):
        for group, indices in [("acoustic", [1, 2]), ("semantic", [3, 4]),
                               ("combined", [1, 2, 3, 4]), ("answer", [5, 6])]:
            with self.subTest(group=group):
                hidden = Hidden([[0, 0] for _ in range(7)])
                cache = object()
                hook, state = make_steering_hook(Vector([2, 4]), None, "fixed", group,
                                                  0.5, self.bounds, 2)
                changed, saved_cache = hook(None, (), (hidden, cache))
                for i in range(7):
                    self.assertEqual(changed.rows[i], [1, 2] if i in indices else [0, 0])
                    self.assertEqual(hidden.rows[i], [0, 0])
                self.assertIs(saved_cache, cache)
                self.assertEqual(state, {"forwards": 1, "steered": 1})

    def test_per_pass_uses_matching_vector_and_skips_missing_measurements(self):
        direction = [Vector([2, 4]), Vector([6, 8])]
        hook, state = make_steering_hook(direction, [True, False], "per-pass", "answer",
                                         -1, self.bounds, 2)
        hidden = Hidden([[0, 0] for _ in range(7)])
        changed, _ = hook(None, (), (hidden, None))
        self.assertEqual(changed.rows[-1], [-2, -4])
        self.assertIsNone(hook(None, (), (hidden, None)))
        self.assertIsNone(hook(None, (), (hidden, None)))
        self.assertEqual(state, {"forwards": 3, "steered": 1})
        fresh, fresh_state = make_steering_hook(direction, [True, False], "per-pass", "answer",
                                               -1, self.bounds, 2)
        self.assertIsNotNone(fresh(None, (), (hidden, None)))
        self.assertEqual(fresh_state["forwards"], 1)

    def test_zero_strength_is_identity(self):
        hook, state = make_steering_hook(Vector([2, 4]), None, "fixed", "answer", 0, {}, 2)
        self.assertIsNone(hook(None, (), (Hidden([[0, 0]]), None)))
        self.assertEqual(state["steered"], 0)

    def test_selection_excludes_extraction_questions_long_clips_and_other_voices(self):
        def row(q, **kwargs):
            return dict(question_id=q, style=kwargs.get("style", "neutral"),
                        speaker=kwargs.get("speaker", "Ryan"), repeat=0,
                        duration_seconds=kwargs.get("seconds", 10))
        rows = [row("train"), row("long", seconds=31), row("other", speaker="Aiden"),
                row("angry", style="angry"), row("new1"), row("new2"), row("new1")]
        selected = select_neutrals(rows, {"train"}, 2, 42)
        self.assertEqual({r["question_id"] for r in selected}, {"new1", "new2"})
        self.assertEqual(selected, select_neutrals(rows, {"train"}, 2, 42))
        with self.assertRaisesRegex(ValueError, "only 2 available"):
            select_neutrals(rows, {"train"}, 3, 42)

    def test_baseline_only_once_across_layers_and_groups(self):
        conditions = make_conditions([10, 15], ["acoustic", "answer"], [0, -0.5, 0.5])
        self.assertEqual(len(conditions), 9)
        self.assertEqual(sum(c["name"] == "baseline" for c in conditions), 1)

    def test_exact_exclusions_match_both_pass_rule_and_neutral_partner(self):
        records, labels = [], []
        for q, neutral_label, angry_label in [("used", "yes", "no"),
                                               ("neutral-failed", "no", "no"),
                                               ("style-failed", "yes", "yes")]:
            for style, label in [("neutral", neutral_label), ("angry", angry_label)]:
                clip = f"{q}-{style}"
                records.append(dict(clip_id=clip, question_id=q, style=style, speaker="Ryan", repeat=0))
                labels.append(dict(clip_id=clip, style=style, predicted_style=label))
        questions, pairs = extraction_questions(records, labels)
        self.assertEqual(questions, {"used"})
        self.assertEqual(pairs, {"angry": 1})

    def test_requested_200_question_sweep_has_23_conditions(self):
        conditions = make_conditions(list(range(15, 26)), ["acoustic", "answer"], [0, 0.5])
        self.assertEqual(len(conditions), 23)
        self.assertEqual({c["layer"] for c in conditions[1:]}, set(range(15, 26)))
        self.assertEqual(len(conditions) * 200, 4600)

    def test_reusing_previous_questions_preserves_order_despite_new_crops(self):
        previous = [dict(clip_id=q, question_id=q) for q in ["q2", "q1"]]
        records = {r["clip_id"]: r for r in previous}
        records["new"] = dict(clip_id="new", question_id="new")
        self.assertEqual(reuse_questions(previous, records, set(), 2), previous)
        with self.assertRaisesRegex(ValueError, "now excluded"):
            reuse_questions(previous, records, {"q1"}, 2)
        with self.assertRaisesRegex(ValueError, "requested 3"):
            reuse_questions(previous, records, set(), 3)

    def test_summary_matches_questions_and_uses_latest_result(self):
        conditions = make_conditions([15], ["answer"], [0, 1])
        name = conditions[1]["name"]
        rows = []
        for q in ["q1", "q2", "q3"]:
            rows += [dict(clip_id=q, condition="baseline", correct=False),
                     dict(clip_id=q, condition=name, correct=True)]
        rows += [dict(clip_id="no-baseline", condition=name, correct=True),
                 dict(clip_id="q1", condition=name, correct=False),
                 dict(clip_id="q1", condition=name, correct=True)]
        with redirect_stdout(io.StringIO()):
            summary = summarize(rows, conditions)[0]
        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["wins"], 3)
        self.assertEqual(summary["losses"], 0)
        self.assertEqual(summary["mcnemar_exact_p"], 0.25)
