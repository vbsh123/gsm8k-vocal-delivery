# CPU-only pitch checks

The earlier speed sweep used `librosa.effects.time_stretch` at a fixed 16-kHz sample rate. This is intended to preserve pitch; it does not guarantee artifact-free speech. These scripts measure fundamental frequency (F0) on voiced frames with pYIN. They never load DIFFA or use the GPU.

## Listen to and measure the old speed experiment

Start with three questions and every speed from the original saved run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv-diffa/bin/python -u audit_speed_pitch.py
```

Download `outputs/spoken-gsm-speed100-pitch-audit` and open its `index.html`, keeping the `audio` subfolder beside it. The players contain unchanged and stretched versions of the same question. Listen for pitch changes, lost words and metallic/smeared artifacts.

Saved inference waveforms are preferred when the old run used `--save-audio`. Otherwise the script reconstructs the transform with the recorded librosa version and checks the original file fingerprint. Such a reconstruction tests the recorded pipeline; it cannot verify an inference waveform that was never archived. The listening page labels the distinction.

To extend the pitch measurement to the whole old 100-question sweep while retaining the three-question listening set:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv-diffa/bin/python -u audit_speed_pitch.py --num-examples 100
```

It resumes completed measurements. `--summary-only` prints saved results. Use another `--output` with `--speed-run outputs/spoken-gsm-speed100-extended` to audit the new experiment separately.

The report compares per-clip mean and median F0 against 1× and reports a time-aligned pitch-contour error. The alignment uses the known stretch factor and is approximate near phoneme/voicing boundaries. Small average changes do not exclude local distortions or pitch-tracking errors.

## Compare emotional delivery against neutral

Start with 20 question/voice groups to check runtime and estimator behavior:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv-diffa/bin/python -u measure_audio_pitch.py --evaluated-only --limit-questions 20
```

Omit `--limit-questions 20` to continue over all completed DIFFA-evaluated Azure clips. Defaults read the first emotional/neutral Jenny folders and `azure-jenny-next500`. `--paths` can select other folders. No correctness or emotion-classifier filter is applied. Requested deliveries are compared on matching question ID and voice, and each recording/question gets equal weight rather than giving longer recordings more weight.

`outputs/jenny-pitch/pitch.csv` contains mean/median F0 in Hz, voiced fraction, pitch range, source paths and completed math correctness when available. `summary.json` contains matched changes relative to neutral. The terminal also shows per-delivery summaries. `--summary-only` reads saved measurements without audio processing. Initial pYIN/Numba compilation can take time; subsequent clips reuse the compiled estimator.

Silence/unvoiced frames are excluded, never treated as 0-Hz pitch. The estimator searches 50–800 Hz with a 20-ms hop; octave errors, low voiced coverage and unusual voices need listening inspection. Mean pitch measures vocal fundamental frequency, not timbre or all aspects of perceived pitch.

Pitch, intonation, rate and intensity are components of emotional delivery. A pitch association alone cannot show that pitch explains an accuracy difference. A later control could shift neutral audio's pitch while preserving duration and words, or match pitch distributions across deliveries. The existing speed results alone did not rule out rate as a contributor.
