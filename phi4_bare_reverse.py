"""Fit three own-task Phi directions on bare audio and test reverse L11 answer steering."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import mean, median
import time

import torch
import emotion_block_night as night
import phi4_all_layer_diagnostic as hooks
import phi4_gsm_bare_eval as gsm
import phi4_more_bare_eval as more
from kimi_bbh_layer_diagnostic import atomic_save
from steer import read_rows
import winning_diffa_sweeps as winners

PROTOCOL = 'phi4-bare-random-pairs-three-captures-L11-reverse-v1'
SIZES = {'gsm8k': 425, 'bbh5': 125, 'multiarith': 290}
PROMPTS = dict(more.PROMPTS, gsm8k=gsm.PROMPT)
ROOT = Path('outputs/phi4-bare-reverse')
LABEL = 'neutral-L11-answer-minus1'
INTERVENTION = dict(layer=11, group='answer', strength=-1.)


def prepare(task, folder):
    config = folder/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if (plan['protocol'] != PROTOCOL or plan['task'] != task or plan['prompt'] != PROMPTS[task]
                or plan['recipe'] != gsm.RECIPE or plan['intervention'] != INTERVENTION):
            raise ValueError('Frozen reverse-steering settings differ.')
        return plan
    if task == 'gsm8k':
        bank, _, rows, source = gsm.select()
    else:
        bank, _, rows, source = more.select(task, more.TOTALS[task])
    size = SIZES[task]
    order = sorted(bank)
    random.Random(44).shuffle(order)
    fit, test = order[:size], order[size:2*size]
    if len(test) != size or len(set(fit + test)) != 2*size:
        raise ValueError('Expected two disjoint question sets of the specified size.')
    records = {(r['question_id'],r['style']):r for r in rows}
    plan = dict(protocol=PROTOCOL, model='phi', task=task, prompt=PROMPTS[task], recipe=gsm.RECIPE,
                max_new_tokens=512, seed=42, split_seed=44, collection_ids=fit, evaluation_ids=test,
                collection=[{s:records[q,s] for s in gsm.STYLES} for q in fit],
                evaluation=[records[q,'neutral'] for q in test], bank={q:bank[q] for q in fit+test},
                capture_targets=night.PHI_TARGETS, intervention=INTERVENTION, source=source,
                vector_definition='Mean of per-question angry minus neutral native hidden states. '
                    'L0 audio prefill, L11 and L30 last prediction position averaged over all forwards per delivery.')
    winners.freeze(config,plan)
    return plan


def fit_vectors(folder, plan, native):
    config_hash = gsm.sha(folder/'run_config.json')
    state = night.load_state(plan,folder)
    for i, pair in enumerate(plan['collection'][len(state['completed_ids']):],len(state['completed_ids'])+1):
        question = pair['neutral']['question_id']
        traces = {}
        for style in gsm.STYLES:
            trace = hooks.generate(native,pair[style],plan['prompt'],collect=True,
                                   capture_layers=[0,11,30],max_new_tokens=512)
            means = trace.pop('means')
            if tuple(means.shape) != (3,2,3072):
                raise ValueError('Unexpected Phi capture dimensions.')
            trace['selected_means'] = torch.stack((means[0,1],means[1,0],means[2,0]))
            trace['forwards'] = trace['hook_forwards']
            traces[style] = trace
        night.append_pair(plan,state,question,traces['neutral'],traces['angry'])
        atomic_save(folder/'collection-stream.pt',state)
        print(f'Fit {i}/{len(plan["collection"])} {question}: tokens '
              f'{traces["neutral"]["response_tokens"]} neutral / {traces["angry"]["response_tokens"]} angry.',flush=True)
    if state['completed_ids'] != plan['collection_ids']:
        raise ValueError('Fitting IDs differ from the frozen cohort.')
    directions = state['difference_sum'].div(len(state['completed_ids'])).float()
    path = folder/'style-vectors.pt'
    named = dict(zip(night.PHI_TARGETS,directions))
    if path.exists():
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if (saved['run_config_sha256'] != config_hash or saved['collection_ids'] != plan['collection_ids']
                or any(not torch.equal(saved['directions'][k],v) for k,v in named.items())):
            raise ValueError('Saved vectors differ from fitting checkpoint.')
    else:
        atomic_save(path,dict(protocol=PROTOCOL,run_config_sha256=config_hash,collection_ids=plan['collection_ids'],
                              sign='angry minus neutral', directions=named, targets=night.PHI_TARGETS))
    return named['L11-answer']


def saved_answers(folder,plan,repair=False):
    records = {r['question_id']:r for r in plan['evaluation']}
    checksum = gsm.sha(folder/'run_config.json')
    vector = folder/'style-vectors.pt'
    vector_hash = gsm.sha(vector) if vector.exists() else None
    done = {}
    for row in read_rows(folder/'results.jsonl',repair=repair):
        q, c = row['question_id'], row['condition']
        r = records.get(q)
        steered = c == LABEL
        if (r is None or c not in ('neutral', LABEL) or (q,c) in done
                or row['run_config_sha256'] != checksum or row['prompt'] != plan['prompt']
                or row['recipe'] != plan['recipe'] or row['seed'] != 42 or row['max_new_tokens'] != 512
                or row['style'] != 'neutral' or any(row[k] != r[k] for k in ('voice','audio_sha256','reference_answer'))
                or row['interventions'] != ([INTERVENTION] if steered else [])
                or row['strength'] != (-1. if steered else 0.) or (steered and not vector_hash)
                or row['vector_sha256'] != (vector_hash if steered else None)):
            raise ValueError(f'Unexpected or duplicate held-out result: {q}/{c}')
        done[q,c] = row
    return done


def pilot_baselines(plan):
    task = plan['task']
    folder = Path('outputs/phi4-gsm-bare-eval100') if task == 'gsm8k' else Path('outputs/phi4-bare-eval100')/task
    config = folder/'run_config.json'
    if not config.exists():
        return {}
    pilot = json.loads(config.read_text())
    if (pilot.get('task') != task or pilot.get('prompt') != plan['prompt']
            or pilot.get('recipe') != plan['recipe']):
        return {}
    reader = gsm.answers if task == 'gsm8k' else more.answers
    rows = reader(folder,pilot)
    source_hash = gsm.sha(config)
    matches = {}
    for record in plan['evaluation']:
        q = record['question_id']
        row = rows.get((q,'neutral'))
        if row and all(row[k] == record[k] for k in ('audio_sha256','reference_answer','voice','style')):
            matches[q] = dict(response=row['response'],response_tokens=row['response_tokens'],
                              reused_from=str(folder/'results.jsonl'),source_config_sha256=source_hash)
    return matches


def evaluate(folder,plan,native):
    direction = fit_vectors(folder,plan,native)
    vector_hash = gsm.sha(folder/'style-vectors.pt')
    checksum = gsm.sha(folder/'run_config.json')
    done = saved_answers(folder,plan,repair=True)
    cached = pilot_baselines(plan)
    with (folder/'results.jsonl').open('a') as stream:
        for i,record in enumerate(plan['evaluation'],1):
            q = record['question_id']
            for c in ('neutral',LABEL):
                if (q,c) in done:
                    continue
                steering = [INTERVENTION] if c == LABEL else []
                if not steering and q in cached:
                    trace = cached[q]
                else:
                    trace = hooks.generate(native,record,plan['prompt'],max_new_tokens=512,
                        interventions=[dict(item,direction=direction) for item in steering])
                row = dict(record,**trace,condition=c,strength=-1. if steering else 0.,
                           run_config_sha256=checksum,vector_sha256=vector_hash if steering else None,
                           prompt=plan['prompt'],recipe=plan['recipe'],seed=42,max_new_tokens=512,interventions=steering)
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                print(f'Eval {i}/{len(plan["evaluation"])} {q}/{c}: saved {row["response_tokens"]} tokens.',flush=True)


def report(folder,plan):
    state = night.load_state(plan,folder)
    rows = saved_answers(folder,plan)
    ids = [q for q in plan['evaluation_ids'] if (q,'neutral') in rows and (q,LABEL) in rows]
    print(f'Phi {plan["task"]}: {len(state["completed_ids"])}/{len(plan["collection_ids"])} fit pairs, '
          f'{len(ids)}/{len(plan["evaluation_ids"])} held-out neutral recordings scored in both conditions.',flush=True)
    if not ids:
        return
    from rescore_audio_steering import paired
    paired_rows = [rows[q,c] for q in ids for c in ('neutral',LABEL)]
    scored, provenance = winners.score_rows(dict(target=plan),paired_rows,folder/'scoring')
    baseline, reverse = scored[::2], scored[1::2]
    lengths, metrics = {}, {}
    for c, group in [('neutral',baseline),(LABEL,reverse)]:
        tokens = [r['response_tokens'] for r in group]
        lengths[c] = dict(mean=mean(tokens),median=median(tokens))
        print(f'{c}: mean/median tokens={mean(tokens):.1f}/{median(tokens):g}')
    for m in baseline[0]['metrics']:
        s = paired([(a['metrics'][m],b['metrics'][m]) for a,b in zip(baseline,reverse)])
        metrics[m] = s
        print(f'{m}: {s["baseline_correct"]/len(ids):.1%} -> {s["steered_correct"]/len(ids):.1%}; '
              f'gain={s["gain_pp"]:+.2f} pp; p={s["mcnemar_exact_p"]:.4g}; CI={s["ci95_gain_pp"]}')
    winners.atomic_text(folder/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(folder/'summary.json',json.dumps(dict(paired=metrics,lengths=lengths,matched=len(ids),
                        scorer=provenance,run_config_sha256=gsm.sha(folder/'run_config.json')),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=SIZES,required=True)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    args = parser.parse_args()
    folder = ROOT/args.dataset
    if args.mode == 'summary':
        if (folder/'run_config.json').exists():
            report(folder,json.loads((folder/'run_config.json').read_text()))
        else:
            print(f'{args.dataset}: awaiting split preparation.')
        return
    with more.dataset_locks(ROOT,[args.dataset],concurrent=True):
        plan = prepare(args.dataset,folder)
        print(f'{args.dataset}: {len(plan["collection_ids"])} fitting pairs; {len(plan["evaluation"])} held-out. '
              f'Neutral vs L11 reverse x1. Prompt: {plan["prompt"]}',flush=True)
        if args.mode == 'prepare':
            winners.score_rows(dict(target=plan),[],folder/'scoring-preflight')
            return
        if len(saved_answers(folder,plan,repair=True)) == 2*len(plan['evaluation']):
            return
        while True:
            # If another dataset is using the remaining VRAM, retry admission only.
            try:
                with more.model_session(concurrent=True,budget=15.) as native:
                    winners.freeze(folder/'runtime.json',dict(model=plan['recipe'],runner_sha256=gsm.sha(__file__),
                        native_hooks_sha256=gsm.sha(hooks.__file__),torch=version('torch'),transformers=version('transformers')))
                    evaluate(folder,plan,native)
                return
            except RuntimeError as exc:
                if 'GiB free for concurrent startup' not in str(exc):
                    raise
                print(f'{exc} Retrying GPU admission in 30 seconds.',flush=True)
                time.sleep(30)


if __name__ == '__main__':
    main()
