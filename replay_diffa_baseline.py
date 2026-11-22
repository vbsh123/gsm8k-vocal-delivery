"""Replay a small frozen neutral sample on the new machine; no collection or steering."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('outputs/diffa-gsm-shared-three-zeroshot-hashes/angry/primary'))
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--run', action='store_true', help='Load DIFFA and replay the selected cases on the GPU.')
    parser.add_argument('--output', type=Path, default=Path('outputs/new-machine-neutral-replay'))
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    import diffa_happy_nightly as task
    from steer import read_rows
    job = json.loads((args.source/'run_config.json').read_text())
    target = job['target']
    records = {r['question_id']: r for r in target['records']}
    saved = {}
    for row in read_rows(args.source/'results.jsonl'):
        if row.get('condition') != 'neutral' or row.get('interventions'):
            continue
        q = row['question_id']
        if q in records and task.winners.reusable(row, job, records[q], target):
            saved[q] = row
    if not saved:
        raise ValueError('No compatible saved neutral baselines found in the restored bundle.')
    ids = sorted(saved)
    random.Random(42).shuffle(ids)
    longest = max(ids, key=lambda q: records[q].get('seconds', 0))
    chosen = ([longest] + [q for q in ids if q != longest])[:args.count]
    print(f'Replay {len(chosen)} frozen neutral cases, including the longest available clip. No steering.', flush=True)
    for q in chosen:
        print(q, f'{records[q].get("seconds", 0):.1f}s', flush=True)
    if not args.run:
        print('Selection only. Add --run on the NEW machine to run the GPU replay.')
        return
    import fcntl
    import subprocess
    import time
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = dict(max_new_tokens=target['max_new_tokens'])
    with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
        print('Waiting for shared GPU lock and idle GPU.', flush=True)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            time.sleep(30)
        from extract_vectors import load_model
        model, tokenizer, features = load_model()
        results = []
        for index, q in enumerate(chosen, 1):
            response, encoding = task.generate(records[q], recipe, target['prompt'], model, tokenizer, features)
            match = response == saved[q]['response']
            print(f'[{index}/{len(chosen)}] {q}: {"IDENTICAL" if match else "DIFFERENT"}', flush=True)
            results.append(dict(question_id=q, identical_response=match, expected=saved[q]['response'],
                                actual=response, audio_sha256=records[q]['audio_sha256'], audio_encoding=encoding))
            (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    mismatches = [r['question_id'] for r in results if not r['identical_response']]
    print(f'Identical responses: {len(results)-len(mismatches)}/{len(results)}. Details: {args.output/"results.json"}')
    if mismatches:
        raise SystemExit('Replay differs. Check software/backend differences before reusing old-machine baselines.')


if __name__ == '__main__':
    main()
