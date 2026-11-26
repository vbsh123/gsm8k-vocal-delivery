"""Evaluate Kimi on the original neutral and angry spoken IFEval recordings."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import kimi_bbh_layer_diagnostic as kimi
import run_spoken_ifeval as official
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from steer import read_rows
from winning_diffa_sweeps import atomic_text, digest, freeze

OUTPUT = Path('outputs/kimi-spoken-ifeval508')
SPLIT = Path('configs/ifeval-spoken508-split.json')
STYLES = ('neutral', 'angry')
# Sorted (question_id, requested style, WAV SHA256) from the original DIFFA run_config.
AUDIO_SHA256 = '9d058255254931d73868b0acca0c628396efc24e534f5e23579e5890f40b96eb'


def prepare(source):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    if not (source/'plan.jsonl').is_file():
        raise FileNotFoundError(f'Missing {source}/plan.jsonl. Copy outputs/azure-spoken-ifeval from the DIFFA machine '
                                'or set IFEVAL_AUDIO_DIR to the original recordings directory.')
    split = json.loads(SPLIT.read_text())
    ids = split['collection_ids'] + split['test_ids']
    if len(ids) != 508 or len(set(ids)) != 508 or split['official_revision'] != official.REVISION:
        raise ValueError('Unexpected spoken IFEval split')
    official_data = official.setup_official()/'data/input_data.jsonl'
    examples = {f"ifeval_{r['key']}":r for r in read_rows(official_data)}
    bank = {q:examples[q] for q in ids}
    lookup = {}
    for r in read_rows(source/'plan.jsonl'):
        q, style = r['question_id'], r['style']
        if q not in bank or style not in STYLES or r['provider'] != 'azure':
            continue
        if (q, style) in lookup or r['question'] != bank[q]['prompt'] or r['voice'] != 'en-US-JennyNeural':
            raise ValueError(f'Mismatched recording for {q}/{style}')
        path = source/r['audio_path']
        lookup[q,style] = dict(question_id=q, condition=style, voice=r['voice'],
                               audio_file=str(path.resolve()), audio_sha256=digest(path))
    records = [lookup[q,s] for q in ids for s in STYLES]
    hashes = sorted((r['question_id'],r['condition'],r['audio_sha256']) for r in records)
    if hashlib.sha256(json.dumps(hashes).encode()).hexdigest() != AUDIO_SHA256:
        raise ValueError('The supplied audio differs from the original DIFFA spoken IFEval recordings.')
    plan = dict(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL,
                tokenizer_revision=TOKENIZER_REVISION, task='ifeval', bank=bank, records=records, seed=42,
                official_revision=official.REVISION, prompt=official.PROMPT, params=dict(kimi.PARAMS, max_new_tokens=1024))
    freeze(OUTPUT/'run_config.json', plan)
    print(f"Verified 508 neutral/angry pairs. Written wrapper: {plan['prompt']}", flush=True)
    return plan


def generate(plan):
    from run_diffa_spoken_prompt import saved
    from run_kimi_multiarith import text_tokens
    from evaluate_kimi_audio import messages
    import torch
    done = saved(OUTPUT, plan, repair=True)
    pending = [r for r in plan['records'] if (r['question_id'], r['condition']) not in done]
    if not pending:
        return
    checksum = digest(OUTPUT/'run_config.json')
    with gpu_queue():
        model = kimi.load_model(Path('external/Kimi-Audio'))
        with torch.inference_mode(), (OUTPUT/'deliveries.jsonl').open('a') as stream:
            for i, r in enumerate(pending, len(done)+1):
                if digest(r['audio_file']) != r['audio_sha256']:
                    raise ValueError(f"Recording changed: {r['audio_file']}")
                kimi.seed()
                _, response = model.generate(messages(r['audio_file'], plan['prompt']), **plan['params'])
                tokens = text_tokens(model, response)
                stream.write(json.dumps(dict(r, response=response, response_tokens=tokens, run_config_sha256=checksum))+'\n')
                stream.flush()
                print(f"{i}/{len(plan['records'])} {r['question_id']} {r['condition']} {tokens} tokens", flush=True)


def report(plan, evaluator):
    from run_diffa_spoken_prompt import saved
    rows = saved(OUTPUT, plan)
    ids = [q for q in plan['bank'] if all((q,s) in rows for s in STYLES)]
    print(f"Kimi spoken IFEval: {len(rows)}/{len(plan['records'])} answers, {len(ids)}/{len(plan['bank'])} matched prompts.")
    if not ids:
        return
    scored = {(q,s):official.score(plan['bank'][q], rows[q,s]['response'], evaluator) for q in ids for s in STYLES}
    lengths, comparisons, instruction = {}, {}, {}
    for s in STYLES:
        values = [rows[q,s]['response_tokens'] for q in ids]
        lengths[s] = dict(mean=mean(values), median=median(values))
        print(f'{s} mean/median tokens: {mean(values):.1f}/{median(values):g}')
    for mode in ('strict','loose'):
        result = paired([(scored[q,'neutral'][mode]['prompt'],scored[q,'angry'][mode]['prompt']) for q in ids])
        comparisons[mode] = result
        n = len(ids)
        a, b = result['baseline_correct'], result['steered_correct']
        print(f"{mode}_prompt: neutral {a}/{n} ({100*a/n:.1f}%) -> angry {b}/{n} ({100*b/n:.1f}%); "
              f"{result['gain_pp']:+.2f} pp, 95% CI {result['ci95_gain_pp']}, exact McNemar p={result['mcnemar_exact_p']:.4g}")
        instruction[mode] = {}
        for s in STYLES:
            bits = [x for q in ids for x in scored[q,s][mode]['instructions']]
            instruction[mode][s] = dict(passed=sum(bits), total=len(bits))
            print(f'{mode} individual instructions / {s}: {sum(bits)}/{len(bits)} ({100*mean(bits):.1f}%)')
    atomic_text(OUTPUT/'official-scored.jsonl',''.join(json.dumps(dict(rows[q,s],official_scores=v))+'\n'
                                                       for (q,s),v in scored.items()))
    atomic_text(OUTPUT/'summary.json',json.dumps(dict(matched=ids,lengths=lengths,comparisons=comparisons,
                instruction=instruction,primary_metric='strict_prompt',scorer=official.REVISION),indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','generate','summary'))
    p.add_argument('--source', type=Path, default=Path('outputs/azure-spoken-ifeval'))
    args = p.parse_args()
    config = OUTPUT/'run_config.json'
    if args.mode == 'summary':
        if not config.exists():
            print('Kimi spoken IFEval is awaiting preparation.')
            return
        official.setup_official()
        report(json.loads(config.read_text()), official.official_scorer())
        return
    with job_lock(OUTPUT):
        if args.mode == 'prepare':
            prepare(args.source)
        else:
            generate(json.loads(config.read_text()))


if __name__ == '__main__':
    main()
