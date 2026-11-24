"""Phi BBH5 and MultiArith: fixed100 questions each, neutral/angry, evaluation only."""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import time

from jenny_bundle_registry import Registry, sha
from phi4_gsm_bare_eval import RECIPE, STYLES, registry_root
from steer import read_rows
import winning_diffa_sweeps as winners

PROTOCOL = 'phi4-bbh5-multiarith-bare-eval100-v1'
TOTALS = dict(bbh5=250, multiarith=580)
PROMPTS = dict(
    bbh5=('Solve the problem in the audio. End your response with a final answer line in the format:\n'
          'The answer is (<letter>).\nReplace <letter> with A, B, C, D, or E.'),
    multiarith=('Solve the math problem in the audio. End your response with a final answer line in the format:\n'
                '#### <number>'))


def select(task, count=100, source=None, audio_root=None, *, sample_seed=None):
    if task not in TOTALS or not 1 <= count <= TOTALS[task]: raise ValueError('Invalid task/question count.')
    if source is None:
        source = next((p for p in (Path('outputs/emotion-block-night')/model/task/'run_config.json'
                                  for model in ('phi','diffa')) if p.exists()), None)
    if source is not None:
        parent = json.loads(source.read_text()); bank = parent['bank']
        if parent.get('task') != task or len(bank) != TOTALS[task]:
            raise ValueError(f'Use the complete {task} night plan, not a fitting/held-out subset.')
        order = sorted(bank); random.Random(42).shuffle(order)
        if parent['question_order'] != order: raise ValueError('Source order differs from the fixed seed42 night order.')
        order = order[:count] if sample_seed is None else random.Random(sample_seed).sample(sorted(bank), count)
        bank = {q:bank[q] for q in order}; records = {}
        for row in parent['records']:
            key = row['question_id'],row.get('condition',row.get('style'))
            if key[0] not in bank or key[1] not in STYLES: continue
            if key in records or row['style'] != key[1]: raise ValueError('Duplicate/inconsistent source record.')
            records[key] = dict(row, condition=key[1])
        if set(records) != {(q,s) for q in order for s in STYLES}: raise ValueError('Missing selected neutral/angry audio pair.')
        if audio_root is not None or any(not Path(r['audio_file']).is_file() for r in records.values()):
            registry = Registry(registry_root(audio_root))
            records = {k:dict(registry.relocate(r), condition=k[1]) for k,r in records.items()}
        receipt = dict(source_config=str(source.absolute()),source_config_sha256=sha(source))
    else:
        registry = Registry(registry_root(audio_root))
        order = sorted(registry.neutral_ids(task))
        if len(order) != TOTALS[task]: raise ValueError(f'Expected {TOTALS[task]} {task} questions; found {len(order)}.')
        if sample_seed is None:
            random.Random(42).shuffle(order)
            order = order[:count]
        else:
            order = random.Random(sample_seed).sample(order, count)
        bank, records = {}, {}
        for q in order:
            neutral = registry.choose(task,q,'neutral')
            bank[q] = dict(question_id=q,question=neutral['question'],answer=neutral['answer'])
            for s in STYLES:
                choices = [x for x in registry.options[task,q,s]
                           if x['question'] == neutral['question'] and x['answer'] == neutral['answer']]
                if len({x['record']['audio_sha256'] for x in choices}) != 1:
                    raise ValueError(f'Missing/ambiguous {task}/{q}/{s}; restore the same Jenny audio.')
                records[q,s] = dict(choices[0]['record'],condition=s)
        receipt = dict(audio_root=str(registry.root))
    if task == 'bbh5':
        from diffa_vanilla_all import scoring_bank
        bank = scoring_bank(task,bank,records)
    for row in records.values():
        answer = str(bank[row['question_id']]['answer']).rsplit('####',1)[-1].strip()
        if (row['voice'] != 'en-US-JennyNeural' or row['reference_answer'] != answer
                or sha(row['audio_file']) != row['audio_sha256']): raise ValueError('Source recording/reference changed.')
    return bank, order, [records[q,s] for q in order for s in STYLES], receipt


def validate(plan):
    task = plan.get('task')
    if task == 'multistep_arithmetic_two':
        from run_arithmetic_audio_pilot import BARE_PROMPT, REASONING_PROMPT
        if (plan.get('protocol') != 'phi4-bbh-arithmetic50-v1' or plan.get('recipe') != RECIPE
                or plan.get('prompt') not in (BARE_PROMPT, REASONING_PROMPT) or plan.get('seed') != 42
                or plan.get('max_new_tokens') != 512 or plan.get('strength') != 0. or plan.get('interventions') != []):
            raise ValueError('Frozen Phi BBH arithmetic settings differ.')
        return
    if (task not in TOTALS or plan.get('protocol') != PROTOCOL or plan.get('recipe') != RECIPE
            or plan.get('prompt') != PROMPTS[task] or plan.get('seed') != 42 or plan.get('max_new_tokens') != 512
            or plan.get('strength') != 0. or plan.get('interventions') != []):
        raise ValueError('Frozen Phi bare task/model/prompt settings differ.')


def prepare(task, output, count=100, source=None, audio_root=None):
    output.mkdir(parents=True,exist_ok=True); config = output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text()); validate(plan)
        if plan['task'] != task or len(plan['question_order']) != count:
            raise ValueError('Frozen task/count differs; use another output folder.')
        if source is not None and plan.get('source_config') != str(source.absolute()):
            raise ValueError('Source changed; use the original command or another output folder.')
        return plan
    bank, order, records, receipt = select(task,count,source,audio_root)
    plan = dict(protocol=PROTOCOL,task=task,model='phi',recipe=RECIPE,prompt=PROMPTS[task],
        seed=42,max_new_tokens=512,num_questions=count,question_order=order,bank=bank,recordings=records,
        styles=list(STYLES),strength=0.,interventions=[],**receipt,
        selection='First N in seed42 shuffled full dataset order; same IDs/audio as the DIFFA night source. '
            'Neutral and angry only. No outcome, emotion-label or duration filtering. No collection, TTS or steering.',
        scoring=('Installed BBH5 strict/flexible saved-answer filters and exact match.' if task == 'bbh5' else
            'Same GSM8K strict/flexible saved-answer filters as DIFFA MultiArith; numeric equivalence/format additional.'))
    winners.freeze(config,plan); winners.atomic_text(output/'prompt.txt',plan['prompt']+'\n')
    return plan


def answers(output,plan,repair=False):
    validate(plan); checksum = sha(output/'run_config.json')
    wanted = {(r['question_id'],r['condition']):r for r in plan['recordings']}; done = {}
    for row in read_rows(output/'results.jsonl',repair=repair):
        key = row['question_id'],row['condition']; record = wanted.get(key)
        if (key in done or record is None or row.get('run_config_sha256') != checksum
                or any(row.get(k) != plan[k] for k in ('prompt','seed','max_new_tokens','strength','interventions'))
                or row.get('model') != RECIPE['model'] or row.get('revision') != RECIPE['revision']
                or row.get('task') != plan['task']
                or any(row.get(k) != record[k] for k in ('voice','style','audio_sha256','reference_answer'))
                or type(row.get('response_tokens')) is not int or row['response_tokens'] < 0):
            raise ValueError('Duplicate/incompatible saved Phi bare answer.')
        done[key] = row
    return done


def generate(output,plan,native):
    import phi4_all_layer_diagnostic as hooks
    done = answers(output,plan,repair=True); checksum = sha(output/'run_config.json')
    with (output/'results.jsonl').open('a') as stream:
        for record in plan['recordings']:
            key = record['question_id'],record['condition']
            if key in done: continue
            print(f'Phi {plan["task"]} {plan.get("prompt_label","bare")} [{len(done)+1}/{len(plan["recordings"])}] {key[0]}/{key[1]}',flush=True)
            trace = hooks.generate(native,record,plan['prompt'],collect=False,interventions=[],max_new_tokens=512)
            row = dict(record,**trace,task=plan['task'],prompt=plan['prompt'],seed=42,max_new_tokens=512,
                model=RECIPE['model'],revision=RECIPE['revision'],strength=0.,interventions=[],run_config_sha256=checksum)
            stream.write(json.dumps(row)+'\n'); stream.flush(); done[key] = row
            print(f'  Saved tokens={row["response_tokens"]}; elapsed={row["generation_seconds"]:.1f}s',flush=True)


def report(output,plan):
    from rescore_audio_steering import paired
    raw = list(answers(output,plan).values())
    scored, provenance = winners.score_rows(dict(target=dict(task=plan['task'],bank=plan['bank'])),raw,output/'scoring')
    groups = {s:{r['question_id']:r for r in scored if r['condition'] == s} for s in STYLES}
    ids = sorted(groups['neutral'].keys() & groups['angry'].keys()); lengths,comparisons = {},[]
    print(f'\nPhi {plan["task"]} {plan.get("prompt_label","bare")}: {len(raw)}/{len(plan["recordings"])} answers; '
        f'{len(ids)}/{len(plan["question_order"])} matched neutral/angry pairs.',flush=True)
    for s in STYLES:
        if not ids: continue
        tokens = [groups[s][q]['response_tokens'] for q in ids]
        lengths[s] = dict(n=len(ids),mean_tokens=mean(tokens),median_tokens=median(tokens))
        print(f'{s}: SAME N={len(ids)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}',flush=True)
        for metric in groups[s][ids[0]]['metrics']:
            count = sum(groups[s][q]['metrics'][metric] for q in ids)
            print(f'  {metric}: {count}/{len(ids)} ({count/len(ids):.1%})',flush=True)
    if ids:
        for metric in groups['neutral'][ids[0]]['metrics']:
            stats = paired([(groups['neutral'][q]['metrics'][metric],groups['angry'][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(metric=metric,**stats))
            print(f'angry vs neutral / {metric}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(output/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'summary.json',json.dumps(dict(protocol=plan['protocol'],task=plan['task'],prompt=plan['prompt'],
        matched_questions=len(ids),expected_questions=len(plan['question_order']),lengths=lengths,paired=comparisons,
        scorer=provenance,note='Custom audio prompting; existing upstream saved-answer evaluators. '
            'Exact paired McNemar p and bootstrap95% CI, unadjusted. Evaluation only; no collection or steering.'),indent=2)+'\n')


def summarize_child(task,output):
    cpu = os.environ.get('DIFFA_PYTHON','.venv-diffa/bin/python')
    subprocess.run([cpu,'-u',__file__,'--mode','summary','--datasets',task,'--output',str(output)],check=True)


@contextmanager
def dataset_locks(root,tasks,concurrent=False):
    root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        own = locks.enter_context((root/'runner.lock').open('a'))
        try: fcntl.flock(own,(fcntl.LOCK_SH if concurrent else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Phi bare task queue already active; stop the old combined queue before concurrent workers.')
        for task in sorted(tasks):
            folder = root/task; folder.mkdir(parents=True,exist_ok=True)
            lock = locks.enter_context((folder/'runner.lock').open('a'))
            try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise SystemExit(f'Phi {task} bare evaluation already active.')
        yield


@contextmanager
def model_session(concurrent=False, budget=15.):
    """Concurrent workers serialize startup only, then generate independently."""
    from steer_phi4_gsm_reverse import NativePhi
    Path('outputs').mkdir(exist_ok=True)
    if concurrent:
        from evaluate_bbh_candidates import concurrent_budget_now
        # Recheck free memory after the other worker has loaded its weights.
        with Path('outputs/phi4-concurrent-startup.lock').open('a') as startup:
            fcntl.flock(startup,fcntl.LOCK_EX)
            concurrent_budget_now(budget)
            native = NativePhi(11)
        yield native
    else:
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU; GSM may finish first.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            yield NativePhi(11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=tuple(TOTALS),default=list(TOTALS))
    parser.add_argument('--num-questions',type=int,default=100)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--source',type=Path,help='Complete night config, when selecting one dataset.')
    parser.add_argument('--audio-root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/phi4-bare-eval100'))
    parser.add_argument('--concurrent',action='store_true',help='Share the GPU with another Phi worker; bypass the idle-GPU queue.')
    parser.add_argument('--gpu-budget-gib',type=float,default=15.,help='Concurrent per-worker PyTorch allocator cap; default15 GiB.')
    args = parser.parse_args()
    if args.gpu_budget_gib <= 0: parser.error('GPU budget must be positive.')
    if len(set(args.datasets)) != len(args.datasets): parser.error('Select each dataset once.')
    if args.source and len(args.datasets) != 1: parser.error('--source requires exactly one dataset.')
    if args.mode == 'summary':
        for task in args.datasets:
            folder = args.output/task
            if (folder/'run_config.json').exists(): report(folder,json.loads((folder/'run_config.json').read_text()))
            else: print(f'Phi {task} bare not prepared yet.',flush=True)
        return
    with dataset_locks(args.output,args.datasets,args.concurrent):
        jobs = [(args.output/task,prepare(task,args.output/task,args.num_questions,args.source,args.audio_root)) for task in args.datasets]
        for folder,plan in jobs:
            print(f'Phi {plan["task"]}: {len(plan["question_order"])} fixed questions × neutral/angry; evaluation only.\n'+plan['prompt'],flush=True)
            if args.mode == 'prepare': winners.score_rows(dict(target=dict(task=plan['task'],bank=plan['bank'])),[],folder/'scoring-preflight')
        if args.mode == 'prepare':
            print('Prepared; no TTS, classifier selection, collection, steering or GPU inference.',flush=True); return
        pending = [(folder,plan) for folder,plan in jobs if len(answers(folder,plan)) < len(plan['recordings'])]
        if pending:
            for folder,plan in pending:
                for row in plan['recordings']:
                    if sha(row['audio_file']) != row['audio_sha256']: raise ValueError('Frozen audio missing/changed.')
            with model_session(args.concurrent,args.gpu_budget_gib) as native:
                for folder,plan in jobs:
                    if len(answers(folder,plan)) < len(plan['recordings']):
                        runtime = dict(recipe=RECIPE,runner_sha256=sha(__file__),
                            packages={k:version(k) for k in ('torch','transformers')},model_config=native.model.config.to_dict(),
                            resource_policy=dict(mode='concurrent' if args.concurrent else 'exclusive',
                                gpu_budget_gib=args.gpu_budget_gib if args.concurrent else None,
                                note='Shared-GPU timings are not comparable to isolated runs.'))
                        key = hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()[:16]
                        winners.freeze(folder/'runtime-receipts'/f'{key}.json',runtime)
                        if not (folder/'runtime.json').exists(): winners.freeze(folder/'runtime.json',runtime)
                        generate(folder,plan,native)
                    summarize_child(plan['task'],args.output)
        else:
            for folder,plan in jobs: summarize_child(plan['task'],args.output)


if __name__ == '__main__': main()
