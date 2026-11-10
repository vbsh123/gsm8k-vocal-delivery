# CPU-only pitch checks

The earlier speed sweep used `librosa.effects.time_stretch` at a fixed 16-kHz sample rate. This is intended to preserve pitch; it does not guarantee artifact-free speech. Pitch measurement uses fundamental frequency (F0) on voiced frames with pYIN. The pitch-analysis and listening scripts never load DIFFA or use the GPU; the optional accuracy stage does.

## Different machine: listening preview, accuracy, then pitch audit

The CPU analysis scripts and listening preview use no GPU. The accuracy stage in `run_speed_checks.sh` does use DIFFA on the GPU.

On a fresh NVIDIA GPU machine:

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/vbsh123/gsm8k-vocal-delivery.git
cd gsm8k-vocal-delivery
bash setup_diffa.sh
mkdir -p outputs
nohup bash run_speed_checks.sh > outputs/speed-checks.log 2>&1 < /dev/null &
tail -f outputs/speed-checks.log
```

If already cloned, use `git pull --ff-only` inside the repository instead of cloning again. Setup needs Python 3.10–3.12. The launcher prepares the original 300 clips if the manifest is absent, and first creates `outputs/speed-listen5.zip`: five questions at each of 16 speeds, with a listening page. Download/extract that ZIP to listen while the accuracy stage runs or waits for the GPU.

It then evaluates 100 recordings <=15 seconds at every speed from 0.5x to 2x. This is at most 1,600 answers including 1x, with compatible previous answers reused if available. A shared lock and GPU-process wait prevent starting alongside an existing worker. New inference input waveforms are saved with `--save-audio`. Model weights download on first inference if missing.

After the GPU process exits, the lock is released and a CPU pitch audit runs on five questions at all 16 speeds. The final ZIP is `outputs/spoken-gsm-speed100-extended-pitch-audit.zip`. Existing reused answers without archived input audio are labelled as reconstructions. Accuracy results are in `outputs/spoken-gsm-speed100-extended`; view them anytime:

```bash
.venv-diffa/bin/python evaluate_audio_speed.py --output outputs/spoken-gsm-speed100-extended --summary-only
```

For listening only, without installing DIFFA/model weights:

```bash
python -m venv .venv-listen
.venv-listen/bin/pip install librosa==0.11.0 soundfile==0.13.1 huggingface-hub pyarrow
.venv-listen/bin/python -u speed_listen5.py
```

This creates `speed-listen5.zip`, downloading public source audio if local originals are absent. The preview consists of new demonstrations rather than archived old-run inputs.

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
