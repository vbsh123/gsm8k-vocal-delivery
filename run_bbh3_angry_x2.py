"""BBH3 bare-task angry block vector at L31 answer x+2; exact existing holdout."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import emotion_block_followups as follow

winners = follow.winners


def prepare(root, output, name='bbh3', strength=2.):
    if name not in ('bbh3', 'bbh5'):
        raise ValueError('Expected BBH3 or BBH5.')
    if strength not in (2., 4.):
        raise ValueError('Expected steering strength x2 or x4.')
    case = follow.complete_case(root, name, emotions=('angry',))['angry']
    target = case['descriptor']
    if target['prompt'] != follow.night.catalog.prompt_for(name):
        raise ValueError(f'Expected the bare-task {name.upper()} prompt and its original block vector.')
    job = dict(protocol=follow.PROTOCOL, kind='strengths', emotion='angry',
        source=target, target=target, strengths=[0., 1., strength],
        layer=31, group='answer', mode='fixed', schedule='always', models=case['models'])
    winners.freeze(output/'run_config.json', job)
    done = follow.validate(job, output, complete_baseline=False, repair=True)
    with (output/'results.jsonl').open('a') as stream:
        for row in case['rows']:
            key = row['question_id'], row['voice'], row['condition']
            if key not in done:
                stream.write(json.dumps(row)+'\n'); done[key] = row
    done = follow.validate(job, output)
    pending = len(target['records'])*len(job['strengths'])-len(done)
    print(f'{name.upper()} angry bare-task vector, L31 answer x{strength:+g}: held-out N={len(target["records"])}; '
          f'{pending} new answers pending. Existing baseline and x+1 reused; no recollection.', flush=True)
    print('Prompt:\n'+target['prompt'], flush=True)
    print('Vector: '+target['vector']+'\nSHA256: '+target['vector_sha256'], flush=True)
    return job


def main(name='bbh3', strength=2.):
    parser = argparse.ArgumentParser(description=f'{name.upper()} bare-task angry block vector at L31 answer x{strength:+g}; exact existing holdout.')
    parser.add_argument('--mode', choices=('run', 'prepare', 'summary'), default='run')
    parser.add_argument('--root', type=Path, default=Path('outputs/emotion-block-night'))
    parser.add_argument('--output', type=Path, default=Path(f'outputs/{name}-angry-bare-x{strength:g}'))
    parser.add_argument('--audio-root', type=Path, action='append')
    args = parser.parse_args()
    config = args.output/'run_config.json'
    if args.mode == 'summary':
        if not config.exists():
            print(f'{name.upper()} angry x{strength:+g} has not been prepared yet.', flush=True); return
        job = json.loads(config.read_text())
        if job['target']['name'] != name or job['strengths'] != [0.,1.,strength]:
            raise ValueError(f'Expected {name} x{strength:+g} results in {args.output}.')
        follow.validate(job, args.output)
        follow.night.vanilla.task.report_steering(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try: fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit(f'{name.upper()} angry x{strength:+g} already active; use --mode summary.')
        job = prepare(args.root, args.output, name, strength)
        if args.mode == 'prepare': return
        done = follow.validate(job, args.output)
        if len(done) == len(job['target']['records'])*len(job['strengths']):
            follow.night.vanilla.task.report_steering(args.output); return
        repo = Path(__file__).resolve().parent
        roots = [*(args.audio_root or [repo/'outputs/jenny-zip-audio']), repo]
        paths, _ = follow.resolve_audio(job, roots)
        if follow.model_revisions() != job['models']:
            raise ValueError('Current DIFFA model revisions differ from the existing block-vector run.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model, tokenizer, features = load_model()
            # Confirm unchanged neutral generation before adding the new intervention.
            original = next(r for r in done.values() if r['strength']==0.)
            record = dict(original, audio_file=paths[original['audio_sha256']])
            response, _ = follow.night.vanilla.task.generate(record, job['target'], job['target']['prompt'],
                model, tokenizer, features)
            if response != original['response']:
                raise ValueError(f'Neutral replay differs; no new x{strength:+g} steering started.')
            winners.generate_job(follow.with_audio_paths(job, paths), args.output, model, tokenizer, features)
            follow.validate(job, args.output)
            follow.night.vanilla.task.report_steering(args.output, tokenizer)


if __name__ == '__main__':
    main()
