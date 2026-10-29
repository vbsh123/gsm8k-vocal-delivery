import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import generate


class FakeBackend:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def generate(self, task, target):
        self.calls.append(task["clip_id"])
        if task["clip_id"] == self.fail:
            raise RuntimeError("simulated interruption")
        # A stand-in for a completed WAV; no fake audio enters the repository.
        target.write_bytes(b"RIFF" + b"\0" * 100)
        return {"duration_seconds": 1.0, "sample_rate": 24000}


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "pilot"
        self.argv = ["--input-jsonl", str(generate.ROOT / "examples/tiny.jsonl"),
                     "--num-examples", "2", "--output", str(self.output)]

    def tearDown(self):
        self.temp.cleanup()

    def quiet_main(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return generate.main(self.argv + ["--prepare-only"] + list(extra))

    def test_all_seven_styles_keep_identical_questions_and_paired_seeds(self):
        self.assertEqual(self.quiet_main(), 0)
        plan = generate.read_jsonl(self.output / "plan.jsonl")
        self.assertEqual(len(plan), 14)
        for question_id in {row["question_id"] for row in plan}:
            rows = [row for row in plan if row["question_id"] == question_id]
            self.assertEqual(len({row["text"] for row in rows}), 1)
            self.assertEqual(len({row["seed"] for row in rows}), 1)
            self.assertEqual({row["style"] for row in rows},
                             {"neutral", "confident", "hesitant", "happy", "angry", "sad", "stressed"})
        stressed = next(row["instruction"] for row in plan if row["style"] == "stressed")
        self.assertIn("psychologically stressed", stressed)

    def test_repeats_and_voices_have_distinct_seeds_but_match_across_styles(self):
        self.assertEqual(self.quiet_main(["--speakers", "Ryan", "Aiden", "--repeats", "2"]), 0)
        plan = generate.read_jsonl(self.output / "plan.jsonl")
        self.assertEqual(len(plan), 56)
        self.assertEqual(len({row["clip_id"] for row in plan}), 56)
        self.assertEqual(len({row["seed"] for row in plan}), 8)

    def test_prepared_styles_use_the_right_words_on_matched_questions(self):
        import subprocess
        import sys

        prepared = Path(self.temp.name) / "prepared"
        subprocess.run([sys.executable, str(generate.ROOT / "prepare_text.py"),
                        "--style", "all", "--num-examples", "2", "--output", str(prepared)],
                       check=True, capture_output=True)
        rows = generate.read_jsonl(prepared / "tts-input.jsonl")
        self.assertEqual(self.quiet_main(["--input-jsonl", str(prepared / "tts-input.jsonl")]), 0)
        plan = generate.read_jsonl(self.output / "plan.jsonl")
        self.assertEqual(len(plan), 14)
        originals = {row["id"]: row for row in rows}
        templates = json.loads((generate.ROOT / "configs/remarks.json").read_text())
        for index, row in enumerate(rows):
            for style, variants in templates.items():
                self.assertEqual(row["style_texts"][style],
                                 variants[index % len(variants)].format(question=row["question"]))
        for task in plan:
            row = originals[task["question_id"]]
            self.assertEqual(task["original_question"], row["question"])
            self.assertEqual(task["text"], row["style_texts"][task["style"]])
            self.assertEqual(task["answer"], row["answer"])
            self.assertNotIn("style_texts", task)

    def test_identical_plan_resumes_but_changed_settings_are_rejected(self):
        self.assertEqual(self.quiet_main(), 0)
        config = (self.output / "run_config.json").read_bytes()
        self.assertEqual(self.quiet_main(), 0)
        self.assertEqual(self.quiet_main(["--temperature", "0.8"]), 1)
        self.assertEqual((self.output / "run_config.json").read_bytes(), config)

    def test_sample_prefix_is_stable_when_increasing_question_count(self):
        args = generate.parser().parse_args(self.argv)
        all_questions, _ = generate.load_questions(args)
        args.num_examples = 1
        smaller, _ = generate.load_questions(args)
        self.assertEqual(smaller, all_questions[:1])

    def test_previous_question_ids_are_excluded(self):
        excluded = Path(self.temp.name) / "previous.jsonl"
        excluded.write_text(json.dumps({"question_id": "demo_apples"}) + "\n")
        self.assertEqual(self.quiet_main(["--exclude-questions", str(excluded), "--num-examples", "1"]), 0)
        questions = generate.read_jsonl(self.output / "questions.jsonl")
        self.assertEqual([q["question_id"] for q in questions], ["demo_books"])

    def test_multiple_previous_batches_are_excluded(self):
        local = Path(self.temp.name) / "questions.jsonl"
        local.write_text(''.join(json.dumps(dict(id=f"q{i}", question=f"Question {i}?", answer="1")) + '\n'
                                 for i in range(4)))
        previous = []
        for index in range(2):
            path = Path(self.temp.name) / f"previous{index}.jsonl"
            path.write_text(json.dumps({"question_id": f"q{index}"}) + '\n')
            previous.append(str(path))
        self.assertEqual(self.quiet_main(["--input-jsonl", str(local), "--num-examples", "2",
                                         "--exclude-questions", *previous]), 0)
        questions = generate.read_jsonl(self.output / "questions.jsonl")
        self.assertEqual({q["question_id"] for q in questions}, {"q2", "q3"})

    def test_missing_exclusion_file_does_not_silently_allow_overlap(self):
        self.assertEqual(self.quiet_main(["--exclude-questions", str(Path(self.temp.name) / "missing.jsonl")]), 1)
        self.assertFalse(self.output.exists())

    def test_failed_clip_is_retried_and_missing_audio_regenerated(self):
        self.assertEqual(self.quiet_main(["--styles", "neutral", "happy"]), 0)
        tasks = generate.read_jsonl(self.output / "plan.jsonl")
        backend = FakeBackend(fail=tasks[1]["clip_id"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(generate.run_tasks(self.output, tasks, backend), 1)
            retry = FakeBackend()
            self.assertEqual(generate.run_tasks(self.output, tasks, retry), 0)
            self.assertEqual(retry.calls, [tasks[1]["clip_id"]])
            (self.output / tasks[0]["audio_path"]).unlink()
            repair = FakeBackend()
            self.assertEqual(generate.run_tasks(self.output, tasks, repair), 0)
            self.assertEqual(repair.calls, [tasks[0]["clip_id"]])
        records = generate.completed_records(self.output)
        self.assertEqual(len(records), 4)
        self.assertTrue(all("audio_sha256" in row for row in records.values()))
        self.assertIn("<audio controls", (self.output / "listen.html").read_text())

    def test_duplicate_styles_and_misspelling(self):
        self.assertEqual(self.quiet_main(["--styles", "hesistant"]), 0)
        plan = generate.read_jsonl(self.output / "plan.jsonl")
        self.assertEqual({row["style"] for row in plan}, {"hesitant"})
        self.assertEqual(self.quiet_main(["--styles", "hesistant", "hesitant"]), 1)

    def test_reference_answer_is_parsed_but_not_in_style_instruction(self):
        self.assertEqual(generate.final_answer("reasoning 6 * 4\n#### 24"), "24")
        self.assertEqual(self.quiet_main(), 0)
        for row in generate.read_jsonl(self.output / "plan.jsonl"):
            self.assertNotIn("Reference answer", row["instruction"])

    def test_unknown_style_and_too_many_questions_fail_before_writing(self):
        self.assertEqual(self.quiet_main(["--styles", "unknown"]), 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.quiet_main(["--num-examples", "99"]), 1)
        self.assertFalse(self.output.exists())

    def test_preview_escapes_question_html(self):
        local = Path(self.temp.name) / "input.jsonl"
        local.write_text(json.dumps({"question": "<script>alert(1)</script>", "answer": "1"}) + "\n")
        self.assertEqual(self.quiet_main(["--input-jsonl", str(local), "--num-examples", "1"]), 0)
        preview = (self.output / "listen.html").read_text()
        self.assertNotIn("<script>", preview)
        self.assertIn("&lt;script&gt;", preview)


if __name__ == "__main__":
    unittest.main()
