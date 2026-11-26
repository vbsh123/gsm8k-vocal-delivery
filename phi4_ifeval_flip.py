"""Replicate the neutral/angry written IFEval flip with native Phi and the saved carrier WAVs."""
import argparse
import json
from pathlib import Path

from phi4_gsm_bare_eval import RECIPE
from phi4_sweep import job_lock
from run_spoken_ifeval import REVISION
import winning_diffa_sweeps as winners

SOURCE = Path('outputs/diffa-ifeval-written-carrier')
OUTPUT = Path('outputs/phi4-ifeval-written-carrier')


def prepare(source, folder):
    config = source / 'run_config.json'
    if not config.is_file():
        raise FileNotFoundError(f'Copy {SOURCE}/run_config.json and its audio/ directory from the DIFFA flip.')
    original = json.loads(config.read_text())
    if (original['protocol'] != 'diffa-ifeval-written-carrier-v1' or original['revision'] != REVISION
            or original['model'] != 'diffa' or original['max_new_tokens'] != 1024):
        raise ValueError('Expected the original 1024-token DIFFA written IFEval flip.')
    bank = original['bank']
    expected = {(q, s) for q in bank for s in ('neutral', 'angry')}
    records = original['records']
    if len(bank) != 541 or len(records) != len(expected) or expected != {
            (r['question_id'], r['condition']) for r in records}:
        raise ValueError('Expected all 541 neutral/angry pairs.')
    paths = {}
    rates = {}
    for r in records:
        q, style, rate = r['question_id'], r['condition'], r['carrier_rate']
        if (r['prompt'] != bank[q]['prompt'] or r['style'] != style or
                r['spoken_prompt'] != 'Here is the task.' or rate != rates.setdefault(q, rate)):
            raise ValueError(f'Inconsistent written prompt, recording or assignment: {q}/{style}.')
        digest = r['audio_sha256']
        if digest not in paths:
            name = Path(r['audio_file']).name
            for path in (source/'audio'/f'rate{rate:+d}'/name, Path(r['audio_file'])):
                if path.is_file() and winners.digest(path) == digest:
                    paths[digest] = path
                    break
            else:
                raise FileNotFoundError(f'Missing original carrier for {style} at rate {rate:+d}% ({name}).')
    plan = dict(original, protocol='phi4-ifeval-written-carrier-v1', model='phi4',
                recipe=dict(RECIPE, max_new_tokens=1024), source_config_sha256=winners.digest(config))
    winners.freeze(folder/'run_config.json', plan)
    return plan, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'summary'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    folder = args.output
    if args.mode == 'summary':
        from ifeval_carrier import report
        config = folder/'run_config.json'
        if config.is_file():
            report(folder, json.loads(config.read_text()), None)
        else:
            print('Phi written IFEval: awaiting preparation.')
        return
    with job_lock(folder):
        plan, paths = prepare(args.source, folder)
        print(f'Phi written IFEval: {len(plan["bank"])} prompts, {len(paths)} verified carrier WAVs.', flush=True)
        if args.mode == 'run':
            from run_multi3_replication import generate_phi, saved
            from phi4_more_bare_eval import model_session
            if len(saved(folder, plan, repair=True)) < len(plan['records']):
                with model_session() as native:
                    generate_phi(folder, plan, native, paths)


if __name__ == '__main__':
    main()
