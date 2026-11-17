"""Freeze completed Phi-4 GSM8K/BBH-three pairs and test an answer-only instruction.

This intervention controls written reasoning requests, not unobservable internal reasoning.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from bbh_candidate_adapters import MODELS
import evaluate_bbh_candidates as candidates
import evaluate_bbh_audio_models as bbh
import evaluate_gsm_candidates as gsm
from kimi_bbh_layer_diagnostic import digest, freeze
from pack_experiment_backup import snapshot
from steer import read_rows

DIRECTIVE = 'Do not show reasoning or explanations. Give only the final answer.'
PROTOCOL = 'phi4-answer-only-frozen-completed-pairs-v1'


def targeted_jobs(root, proc=Path('/proc')):
    """Only the current repository's Phi-4 GSM wrapper/worker and their children."""
    processes = {}; roots = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try:
            if (entry/'cwd').resolve() != root.resolve(): continue
            stat = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] in ('Z', 'X'): continue
            args = [s.decode(errors='replace') for s in (entry/'cmdline').read_bytes().split(b'\0') if s]
            pid = int(entry.name); processes[pid] = dict(parent=int(stat[1]), start=stat[19], args=args)
            names = [Path(a).name for a in args]
            wrapper = 'run_gsm_candidate.sh' in names and args[names.index('run_gsm_candidate.sh')+1:names.index('run_gsm_candidate.sh')+2] == ['phi4']
            worker = ('evaluate_gsm_candidates.py' in names and '--model' in args and
                args[args.index('--model')+1:args.index('--model')+2] == ['phi4'] and '--mode' in args and
                args[args.index('--mode')+1:args.index('--mode')+2] == ['evaluate'])
            if wrapper or worker: roots.add(pid)
        except (OSError, IndexError, ValueError): continue
    while True:
        children = {pid for pid, p in processes.items() if p['parent'] in roots}
        if children <= roots: break
        roots |= children
    return {pid: processes[pid] for pid in sorted(roots)}


def stop_gsm(root, timeout=30):
    jobs = targeted_jobs(root)
    if not jobs:
        print('No active Phi-4 GSM8K queue/worker in this repository.', flush=True); return
    print('Stopping only Phi-4 GSM8K queue/worker PID(s): '+', '.join(map(str, jobs)), flush=True)
    for pid, p in reversed(list(jobs.items())):
        try:
            stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[19] == p['start']: os.kill(pid, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError): pass
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        active = targeted_jobs(root)
        if not active:
            print('Phi-4 GSM8K stopped. Completed responses retained; current unfinished response may be absent.', flush=True); return
        time.sleep(1)
    raise RuntimeError('Phi-4 GSM8K still active after SIGTERM; no cohort was frozen. No SIGKILL issued.')


def answer_only_prompt(task, prompt):
    expected = gsm.ORIGINAL if task == 'gsm8k' else bbh.PROMPT
    if prompt != expected:
        raise ValueError('Expected the ORIGINAL reasoning prompt; choose its run directory.')
    if prompt.count("Let's think step by step.") != 1: raise ValueError('Reasoning instruction is ambiguous.')
    return prompt.replace("Let's think step by step.", DIRECTIVE)


def prepare_task(task, source, output):
    selection_path = output/'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection['source'] != str(source.absolute()) or selection['protocol'] != PROTOCOL:
            raise ValueError('Frozen follow-up settings differ; use a separate output folder.')
        plan = json.loads((output/'answer-only/run_config.json').read_text())
        if plan['prompt'] != answer_only_prompt(task, json.loads((output/'original/run_config.json').read_text())['prompt']):
            raise ValueError('Frozen answer-only prompt differs.')
        print(f'{task}: reusing frozen {len(selection["pairs"])} pairs; source growth does not change this cohort.', flush=True)
        return output/'answer-only', plan
    config = source/'run_config.json'; plan = json.loads(config.read_text())
    if plan['recipe']['model'] != MODELS['phi4']['model'] or plan['recipe']['revision'] != MODELS['phi4']['revision']:
        raise ValueError('Expected the pinned Phi-4 model.')
    prompt = answer_only_prompt(task, plan['prompt'])
    if plan['seed'] != 42 or plan['max_new_tokens'] != 512 or plan['interventions']:
        raise ValueError('Source decoder/seed/interventions differ from this follow-up.')
    data, trimmed = snapshot(source/'results.jsonl')
    rows = [json.loads(s) for s in data.splitlines() if s.strip()]
    tasks = {(r['question_id'], r['voice'], r['style']): r for r in plan['recordings']}
    latest = {}
    for row in rows:
        key = row['question_id'], row['voice'], row['style']
        if key not in tasks or row['prompt'] != plan['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512:
            raise ValueError('Saved baseline row differs from the frozen source configuration.')
        if (row['model'] != plan['recipe']['model'] or row['revision'] != plan['recipe']['revision'] or row['interventions']
                or row['audio_sha256'] != tasks[key]['audio_sha256'] or row['reference_answer'] != tasks[key]['reference_answer']):
            raise ValueError('Saved baseline model, recording or reference differs.')
        if key in latest and latest[key]['response'] != row['response']:
            raise ValueError('Conflicting baseline answers.')
        latest[key] = row
    neutral = {(q, v) for q, v, s in latest if s == 'neutral'}
    angry = {(q, v) for q, v, s in latest if s == 'angry'}
    common = neutral & angry
    if not common: raise ValueError(f'{task}: no completed neutral/angry pairs to freeze.')
    records = [r for r in plan['recordings'] if (r['question_id'], r['voice']) in common]
    pairs = list(dict.fromkeys((r['question_id'], r['voice']) for r in records))
    baseline = [latest[q, v, s] for q, v in pairs for s in ('neutral', 'angry')]
    ids = {q for q, _ in pairs}
    restricted = dict(plan, bank={q: v for q, v in plan['bank'].items() if q in ids}, recordings=records,
        selection='Freeze ALL completed paired answers at follow-up launch; no correctness, format, length or classifier filtering.')
    for folder in (output/'original', output/'answer-only'): folder.mkdir(parents=True, exist_ok=True)
    freeze(output/'original/run_config.json', restricted)
    (output/'original/results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in baseline))
    changed = dict(restricted, protocol=PROTOCOL, prompt=prompt, prompt_variant='answer-only')
    freeze(output/'answer-only/run_config.json', changed)
    freeze(selection_path, dict(protocol=PROTOCOL, source=str(source.absolute()), source_config_sha256=digest(config),
        baseline_snapshot_sha256=hashlib.sha256(data).hexdigest(), incomplete_final_line_omitted=trimmed,
        pairs=[list(p) for p in pairs], baseline_answers=len(baseline), excluded_unpaired_answers=len(latest)-len(baseline),
        caveat='Exploratory follow-up chosen after inspecting interim outcomes. Prompt requests suppressing written reasoning, '
               'not internal reasoning. Baselines and audio are unchanged; primary cohort is never extended on resume.'))
    print(f'{task}: froze {len(pairs)} completed pairs; {len(baseline)} new answer-only generations planned.', flush=True)
    return output/'answer-only', changed


def prepare(args):
    return [prepare_task(task, source, args.output/task) for task, source in
            [('gsm8k', args.gsm_run), ('bbh3', args.bbh_run)]]


def report(args):
    import numpy as np
    from rescore_audio_steering import paired
    for task in ('gsm8k', 'bbh3'):
        root = args.output/task
        if not (root/'selection.json').exists():
            print(f'{task}: not prepared.'); continue
        groups = {}
        for variant in ('original', 'answer-only'):
            folder = root/variant; print(f'\n{task.upper()} / {variant}', flush=True)
            if task == 'gsm8k': gsm.report(folder)
            else: bbh.report(folder)
            path = folder/'rescored.jsonl'
            groups[variant] = {(r['question_id'], r['voice'], r['style']): r for r in read_rows(path)}
        common = set.intersection(*(set(g) for g in groups.values()))
        keys = sorted({(q, v) for q, v, s in common if all((q, v, t) in common for t in ('neutral', 'angry'))})
        if not keys: continue
        metric = 'flexible-numeric' if task == 'gsm8k' else 'flexible-extract'
        def correct(variant, key, style):
            row = groups[variant][(*key, style)]
            return row['scores'][metric] if task == 'gsm8k' else row['harness'][metric]['correct']
        comparisons = []; print(f'\nSame-question prompt comparisons: {task}, N={len(keys)}', flush=True)
        for style in ('neutral', 'angry'):
            result = paired([(correct('original', k, style), correct('answer-only', k, style)) for k in keys])
            comparisons.append(dict(style=style, **result))
            print(f'{style}, answer-only versus original: {result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
        delta = np.array([(int(correct('answer-only', k, 'angry'))-int(correct('answer-only', k, 'neutral')))-
                         (int(correct('original', k, 'angry'))-int(correct('original', k, 'neutral'))) for k in keys])
        values, counts = np.unique(delta, return_counts=True)
        boot = np.random.default_rng(42).multinomial(len(keys), counts/len(keys), 200000)@values/len(keys)*100
        interaction = dict(gain_difference_pp=float(delta.mean()*100), ci95_pp=np.percentile(boot, [2.5,97.5]).tolist())
        print('Change in angry-minus-neutral effect under answer-only:', interaction, flush=True)
        lengths = {v:{s:dict(mean=statistics.mean(groups[v][(*k,s)]['response_tokens'] for k in keys),
            median=statistics.median(groups[v][(*k,s)]['response_tokens'] for k in keys)) for s in ('neutral','angry')} for v in groups}
        print('Mean/median response-token lengths:', lengths, flush=True)
        (root/'comparison.json').write_text(json.dumps(dict(pairs=len(keys), metric=metric, prompt_comparisons=comparisons,
            emotion_by_prompt_interaction=interaction, token_lengths=lengths,
            note='Same frozen questions in all four cells. Exploratory, unadjusted. Written-answer instruction intervention; '
                 'not direct control of internal reasoning and not proof that response length is causal.'), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('stop-gsm','prepare','evaluate','summary'), default='evaluate')
    p.add_argument('--gsm-run', type=Path, default=Path('outputs/gsm8k-candidates/phi4/original'))
    p.add_argument('--bbh-run', type=Path, default=Path('outputs/bbh-three-candidates/phi4/original'))
    p.add_argument('--output', type=Path, default=Path('outputs/phi4-answer-only-followup'))
    args = p.parse_args()
    if args.mode == 'stop-gsm': stop_gsm(Path.cwd()); return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary': report(args); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = prepare(args)
        if args.mode == 'prepare': return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            candidates.run(plans, 'phi4')


if __name__ == '__main__': main()
