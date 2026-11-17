# DIFFA-2 SAME100 reasoning-instruction and emotion vectors

Run this on the **DIFFA / steering machine**, where the existing DIFFA GSM8K
configs, audio, environment and model weights remain available. No Phi-4/Kimi
model is used. The separate Phi-4 collector is retained but not started here.

Exactly the SAME100 question/voice pairs in all three conditions:

1. Neutral audio with reasoning requested.
2. That same neutral audio with an answer-only instruction.
3. Matched angry audio with reasoning requested.

```bash
git pull --ff-only
nohup bash run_diffa_reasoning_emotion100.sh \
  > outputs/diffa-reasoning-emotion100.log 2>&1 < /dev/null &
tail -f outputs/diffa-reasoning-emotion100.log
```

Default source: `outputs/reasoning-transfer/gsm8k/run_config.json`, the existing
aligned-prompt GSM8K evaluation. Select100 seed42-shuffled neutral/angry pairs
from its original unsteered recordings, with no correctness, classifier, format
or duration filter. The source's steered rows are excluded. Its prompt, source
config hash, question IDs and recording hashes are frozen before model loading.
Use `--source-run /path/to/run` to choose an explicit compatible source.

The reasoning sentence is replaced by "Do not show reasoning or explanations.
Give only the final answer." All other wording, including the source's final-
answer format, stays identical. All three conditions use seed42,128 steps,
32-token block setting,512 answer masks, temperature0 and native fast-dllm entry.
The existing `gen_chunked` helper preserves full recordings and takes its native
route for inputs up to30seconds. No new TTS or different model-family weights.

## Capture point

This is **300 initial masked-answer forwards**, not300 full answer generations.
The collector enters the SAME native generation function and intercepts after
the first full LLaDA forward but before token selection/commitment. It captures
all32 block outputs, averaging the same positions used in our DIFFA analysis:
acoustic, semantic, combined audio, and all512 answer-mask positions.

The first masked-answer input is hashed and must be identical across the three
conditions for each question. This prevents different generated histories or
answer lengths from entering the comparison. No activation is modified. Hooks,
audio-insertion interception and the long-audio helper are restored on exit.

DIFFA's decoder is bidirectional: audio-position states CAN depend on the text
instruction. Therefore all four token groups are meaningful to inspect. Phi-4's
causal first-answer capture and its special audio-position caveat do not apply.

## Directions and interpretation

Equal means of100 matched per-question differences:

- reasoning-requested − answer-only, on neutral audio;
- angry − neutral, with the same reasoning request;
- neutral − angry, saved as the exact negative of the preceding direction.

The reverse cosine must have the opposite sign; it is not additional independent
evidence. Per-layer/per-group CSV includes cosine, projection coefficient,
squared cosine overlap, norms and split-half reproducibility. Cross-half checks
compare reasoning from odd selected questions against emotion from even ones and
vice versa, still using only the SAME100 collected questions. They address shared
question-baseline effects descriptively, not as a significance test.

This captures an INITIAL-step **reasoning-instruction** contrast. It is different
from our older steering means averaged over decoding forwards. Similarity can
also reflect wording, expected answer style or prompt-position changes; it does
not establish that anger is reasoning or that the direction causes accuracy gains.

```bash
.venv-diffa/bin/python collect_diffa_reasoning_emotion.py --mode summary
```

Outputs: `outputs/diffa-reasoning-emotion100/` with frozen config, runtime/source
hashes, per-question/per-condition state caches, final `style-vectors.pt`,
`layer-cosines.csv` and `summary.json`. Expect around650MB of state caches for
32×4×4096 float32 summaries; no full encoded inputs or model weights are saved.
Partial collection resumes per condition. Only all100 complete triples produce
the final vector file; partial summaries use a separate filename.

It waits for existing default strength/transfer queues, then the shared GPU lock
and any active GPU work. It never kills another run. As with previous DIFFA jobs,
DIFFA's LLaDA and Whisper weights must exist on this machine or the normal model
loader will need to restore them; don't run this on the Phi-4 machine after
deleting those caches unless you intend to redownload them.
