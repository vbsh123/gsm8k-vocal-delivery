# Emotional spoken IFEval

This is our spoken adaptation of Google's original 541 IFEval prompts, not the
released SpeechInstructBench subset or the separately named IFEval-Audio dataset.
Original prompts and constraint metadata are frozen at Google Research revision
`e6890f85757dd84e27ca6df2dd30651dafad28e0`.

Azure Jenny reads each unchanged prompt in neutral, angry, happy (Azure cheerful),
and sad delivery, intensity 1.5. DIFFA receives audio plus the same generic text
wrapper, "Follow the instructions in the audio." It does not receive the transcript,
emotion label, constraint metadata, or an extra answer-format request.

All four recordings must fit 30 seconds for a question to enter evaluation. No
cropping; excluded IDs and every duration are saved in `eligibility.json`.
Punctuation, markup, URLs and exact-repeat instructions can be difficult to convey
through TTS: listen to `listen.html` before interpreting results as instruction ability.
This is an audio adaptation, not a canonical original-IFEval leaderboard score.

DIFFA uses the existing shared model loader and fast-dLLM generation, 128 steps,
32-token blocks, temperature zero, seed 42 reset for every recording. The default
output budget is 1024 tokens for every style. Long responses can hit that budget;
raw responses and response token counts are saved for inspection. Budget changes
require a separate result folder.

Scoring calls Google's unmodified strict/loose evaluators. Prompt-level success
requires every constraint to pass; instruction-level success counts individual
constraints. These are compliance metrics, not factual correctness judgments.
Prompt-level paired comparisons include McNemar p-values and 200,000-resample
paired bootstrap intervals, without multiple-comparison adjustment.

On the existing evaluation machine, from the repository:

```bash
read -rsp 'Azure speech key: ' AZURE_SPEECH_KEY; echo
export AZURE_SPEECH_KEY
read -rp 'Azure speech region: ' AZURE_SPEECH_REGION
export AZURE_SPEECH_REGION
mkdir -p outputs
nohup bash run_spoken_ifeval.sh > outputs/spoken-ifeval.log 2>&1 < /dev/null &
tail -f outputs/spoken-ifeval.log
```

The shell wrapper installs only the official checker's small Python dependencies
and NLTK tokenizers into the existing DIFFA environment. TTS runs on CPU/cloud;
evaluation then waits for the shared GPU lock and any active GPU process. Model
weights are reused through the existing HF cache. Rerunning resumes saved audio
and answers. No steering is included in this initial four-delivery experiment.

To generate only, for listening first:

```bash
.venv-diffa/bin/python -u run_spoken_ifeval.py --mode generate
```

To inspect results while evaluation continues (CPU only):

```bash
.venv-diffa/bin/python -u run_spoken_ifeval.py --mode summary
```

Main artifacts:

- `outputs/azure-spoken-ifeval/`: original constraint metadata, source hashes,
  request plan, audio, listening page and duration eligibility.
- `outputs/spoken-ifeval-diffa2/`: generation configuration, raw `results.jsonl`,
  separate `official-scored.jsonl`, and `summary.json`.
