# AQuA-RAT audio pilot

50 test questions, selected once by Python `random.Random(42).shuffle` from the
authors' complete 254-question test split, then first 50. No outcome, difficulty,
notation or classifier filtering. Source: [DeepMind AQuA](https://github.com/google-deepmind/AQuA),
revision `26b64ed1f22208c6742f47df547dfd65088ec30d`, test.json Git blob
`a62ef3e8bc0446de096c8384a0bf5c99867043b2`.

Each question and all five choices are spoken with Jenny neutral and angry,
intensity 1.5. 100 WAVs and 100 DIFFA answers. Reference rationales are retained
only in the downloaded original source file; they never enter the question bank,
TTS input or generation prompt. No activation collection or steering.

The same **custom bare audio prompt** is used in both conditions:

```text
Answer the question in the audio. End your response with a final answer line in the format:
The answer is (<letter>).
Replace <letter> with A, B, C, D, or E.
```

The source question/options and spoken transcription are saved together.
Mathematical signs are named explicitly; slash and vertical bars are spoken
literally. Currency and percent are expanded. `5C3` is spoken as “5 choose 3”.
Printed suffix digits such as `a2` are not automatically repaired into powers;
source typos and ambiguities remain, without changing labels or excluding rows.
Listen to clips in `outputs/azure-aqua50/listen.html` before interpreting results.

Generation uses the existing native DIFFA path: seed42, fast_dllm,128 steps,
block32, temperature0,512 maximum new tokens. It waits for the shared GPU lock
and an idle GPU after TTS. Saved WAVs/answers are reused on restart. Audio/config
hashes and runtime model/package receipts are recorded. Every answer logs its
saved response token count and elapsed seconds.

## Scoring

The installed lm-evaluation-harness `agieval_aqua_rat` task scores **choice
likelihoods**, not generated final-answer text. This pilot is explicitly a local
generated-output adaptation, not a reproduction of that likelihood protocol.

- `choice-accuracy`: compare the last standalone `The answer is ...` declaration
  to the original authors' `correct` option. Accept parentheses and Markdown
  wrappers. Do not extract letters from option discussions or arbitrary text.
- `strict-final-line`: correct choice on an exact final line `The answer is (A).`
  through `(E).` with the requested capitalization, punctuation and parentheses.
- `format-compliant`: that exact final-line format, independent of correctness.

Missing/unparseable answers are incorrect. Summary prints matched neutral/angry
accuracy, exact paired McNemar p, paired bootstrap95% CI and matched response
lengths. Tests are exploratory and unadjusted; interim summaries are not new
independent experiments. More tokens do not establish better reasoning.

## Commands

With matching `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` already exported:

```bash
git pull --ff-only
nohup bash run_aqua_audio_pilot.sh > outputs/aqua50.log 2>&1 < /dev/null &
bash run_aqua_audio_pilot.sh --mode summary
python3 show_aqua_outputs.py --count 3
python3 show_aqua_outputs.py --longest --count 3
```

`--mode prepare` validates the source and creates TTS plans without paid calls
or inference. `--mode evaluate` uses the frozen completed audio plan. Re-running
the original run command resumes TTS and missing answers.
