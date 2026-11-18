"""Kimi GSM8K: all 1269 main-cohort questions, random 634/635 split, L17 answer x1."""
import argparse
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import wave

import run_kimi_multiarith as shared
from run_reasoning_transfer import bank_file, generate_audio

PROTOCOL = 'kimi-gsm8k-1269-random634-635-L17-answer-x1-v1'
STEERED = 'neutral-GSM8K-L17-answer-a1'
QUESTION_FILE = Path(__file__).with_name('kimi_gsm8k_questions.json')


def select(source, output):
    source.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTION_FILE.read_text())['questions']
    if len(questions) != 1269 or len({q['question_id'] for q in questions}) != 1269:
        raise ValueError('Expected all 1269 unique main-cohort GSM8K questions.')
    random.Random(42).shuffle(questions)
    train, test = questions[:634], questions[634:]
    bank_file(source/'questions.jsonl', questions)
    selection = dict(protocol=PROTOCOL, dataset='GSM8K', seed=42, collection=train, test=test,
        question_file_sha256=shared.kimi.digest(QUESTION_FILE), prompt=shared.multiarith.PROMPT,
        layer=17, group='answer', strength=1., primary_metric='flexible-numeric',
        primary_comparison='Fixed L17 answer x1 versus neutral on all 635 held-out questions',
        rule='All 1269 questions from three main Azure cohorts; seed42 shuffle, first 634 collection, '
             'remaining 635 test; no outcome/classifier/duration filtering. No old DIFFA vector reused.')
    shared.kimi.freeze(output/'selection.json', selection)
    return selection


def audio(args):
    selection = select(args.source, args.output)
    prepare_audio(args, selection)
    return shared.prepare(args, selection, dataset='GSM8K', steered_condition=STEERED)


def prepare_audio(args, selection, styles=('neutral', 'angry')):
    """Reuse compatible audio or resume TTS for an already frozen question selection."""
    questions = selection['collection']+selection['test']
    # Make the exact resumable request plan before linking compatible existing audio.
    subprocess.run([sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--input-jsonl', str(args.source/'questions.jsonl'), '--num-examples', str(len(questions)),
        '--styles', *styles, '--azure-voice', 'en-US-JennyNeural',
        '--azure-intensity', '1.5', '--output', str(args.source), '--resume', '--prepare-only'], check=True)
    candidates = {}
    for folder in args.audio_sources:
        for row in shared.read_rows(folder/'plan.jsonl'):
            if row.get('provider') == 'azure' and row.get('style') in styles:
                candidates.setdefault((row['question_id'], row['style']), []).append((folder, row))
    linked = 0
    for task in shared.read_rows(args.source/'plan.jsonl'):
        target = args.source/task['audio_path']
        if target.exists():
            continue
        if target.is_symlink():
            target.unlink()  # Only repair a broken link inside this new audio folder.
        for folder, old in candidates.get((task['question_id'], task['style']), []):
            path = (folder/old['audio_path']).absolute()
            if (not path.is_file() or any(old.get(k) != task[k] for k in ('question', 'voice', 'request'))
                    or str(old.get('answer')).rsplit('####', 1)[-1].strip() != str(task['answer'])):
                continue
            try:
                with wave.open(str(path)) as stream:
                    valid = stream.getnframes() > 0
            except (OSError, EOFError, wave.Error):
                continue
            if valid:
                target.symlink_to(path); linked += 1
                break
    print(f'Reused {linked} compatible clips through links; generate/resume remaining clips.', flush=True)
    generate_audio(args.source, questions, list(styles))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'audio', 'evaluate', 'summary'], default='evaluate')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-gsm8k1269'))
    p.add_argument('--output', type=Path, default=Path('outputs/kimi-gsm8k634-635-L17'))
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    p.add_argument('--audio-sources', nargs='+', type=Path, default=[Path('outputs')/s for s in
        ('azure-jenny500-extra', 'azure-jenny500-neutral', 'azure-jenny-next500', 'azure-jenny-remaining-test')])
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        shared.report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            select(args.source, args.output); print('Frozen 634 collection / 635 test; no generation.'); return
        if args.mode == 'audio':
            audio(args); return
        plan = shared.prepare(args, select(args.source, args.output), dataset='GSM8K', steered_condition=STEERED)
        done = {(r['question_id'], r['voice'], r['condition']) for r in shared.read_rows(args.output/'results.jsonl')}
        if all((q['question_id'], q['voice'], c) in done for q in plan['test'] for c in ('neutral', 'angry', STEERED)):
            print('All 1905 test answers saved; no model loading.'); return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            model = shared.kimi.load_model(args.model_source)
            vector = shared.collect(plan, args.output, model)
            shared.evaluate(plan, args.output, model, vector)


if __name__ == '__main__':
    main()
