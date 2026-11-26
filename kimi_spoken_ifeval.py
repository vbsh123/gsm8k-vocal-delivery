"""Evaluate Kimi on the original neutral, angry and happy spoken IFEval recordings."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from statistics import mean, median

import kimi_bbh_layer_diagnostic as kimi
import run_spoken_ifeval as official
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from steer import read_rows
from winning_diffa_sweeps import atomic_text, digest, freeze

PREVIOUS = Path('outputs/kimi-spoken-ifeval508')
OUTPUT = Path('outputs/kimi-spoken-ifeval508-happy')
SPLIT = Path('configs/ifeval-spoken508-split.json')
STYLES = ('neutral', 'angry', 'happy')
# Sorted (question_id, requested style, WAV SHA256) from the archived DIFFA run.
AUDIO_SHA256 = '11f2773055fca302977798d3ec99335dce28788979039e9e0c36aa6df2baa0fc'


def upload_path(source):
    if source is None:
        candidates = [base / name for base in (Path.cwd(), Path('/workspace'), Path.home())
                      for name in ('jenny', 'jenny.zip')]
        candidates.append(Path('outputs/azure-spoken-ifeval'))
        source = next((p for p in candidates if p.exists()), None)
    if source is None or not source.exists():
        raise FileNotFoundError('Supply the Jenny directory or ZIP using IFEVAL_AUDIO_DIR.')
    return source


def audio_folder(source, names):
    anchor = names[0]
    if source.is_dir():
        if (source / anchor).is_file():
            return source
        for relative in ('outputs/azure-spoken-ifeval', 'azure-spoken-ifeval'):
            if (source / relative / anchor).is_file():
                return source / relative
        options = [p.parent for p in source.rglob(anchor) if p.parent.name == 'azure-spoken-ifeval']
        if len(options) == 1:
            return options[0]
        archives = list(source.glob('*.zip'))
        if len(archives) != 1:
            raise ValueError(f'Point IFEVAL_AUDIO_DIR at the IFEval WAV folder or Jenny ZIP inside {source}.')
        source = archives[0]
    if not zipfile.is_zipfile(source):
        raise ValueError(f'{source} is not a readable ZIP. Check that the upload has finished.')
    cache = OUTPUT / 'audio'
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        paths = {Path(n) for n in z.namelist() if Path(n).name == anchor}
        preferred = [p for p in paths if p.parent.name == 'azure-spoken-ifeval']
        roots = preferred or list(paths)
        if len(roots) != 1:
            raise ValueError(f'Cannot uniquely locate IFEval recordings inside {source}.')
        # Write only the selected flat filenames, independently of archive paths.
        for name in names:
            target = cache / name
            temporary = target.with_suffix('.wav.tmp')
            temporary.write_bytes(z.read(str(roots[0].parent / name)))
            temporary.replace(target)
    return cache


def reuse_previous(plan):
    from run_diffa_spoken_prompt import saved
    if not (PREVIOUS / 'deliveries.jsonl').is_file():
        return
    previous = json.loads((PREVIOUS / 'run_config.json').read_text())
    if {k:v for k,v in previous.items() if k != 'records'} != {k:v for k,v in plan.items() if k != 'records'}:
        raise ValueError('Previous Kimi answers have different model, prompts or decoding parameters.')
    earlier = saved(PREVIOUS, previous)
    done = saved(OUTPUT, plan, repair=True)
    checksum = digest(OUTPUT / 'run_config.json')
    reused = 0
    with (OUTPUT / 'deliveries.jsonl').open('a') as stream:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key not in earlier or key in done:
                continue
            row = earlier[key]
            if any(row[k] != v for k,v in record.items() if k != 'audio_file'):
                raise ValueError(f'Previous audio or condition differs for {key}')
            result = dict(row, **record)
            result.update(source_config_sha256=row['run_config_sha256'], run_config_sha256=checksum)
            stream.write(json.dumps(result)+'\n')
            reused += 1
    print(f'Reused {reused} previously completed answers.')


def prepare(source):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    split = json.loads(SPLIT.read_text())
    ids = split['collection_ids'] + split['test_ids']
    if len(ids) != 508 or len(set(ids)) != 508 or split['official_revision'] != official.REVISION:
        raise ValueError('Unexpected spoken IFEval split')
    official_data = official.setup_official()/'data/input_data.jsonl'
    examples = {f"ifeval_{r['key']}":r for r in read_rows(official_data)}
    bank = {q:examples[q] for q in ids}
    # The original TTS exporter numbered the WAVs in official IFEval source order.
    indices = {q:i for i,q in enumerate(examples)}
    names = {(q,s): f'q{indices[q]:05d}-azure-{s}.wav' for q in ids for s in STYLES}
    folder = audio_folder(upload_path(source), list(names.values()))
    records = []
    for (q, style), name in names.items():
        path = folder / name
        records.append(dict(question_id=q, condition=style, voice='en-US-JennyNeural',
                            audio_file=str(path.resolve()), audio_sha256=digest(path)))
    hashes = sorted((r['question_id'],r['condition'],r['audio_sha256']) for r in records)
    if hashlib.sha256(json.dumps(hashes).encode()).hexdigest() != AUDIO_SHA256:
        raise ValueError('The supplied audio differs from the original DIFFA spoken IFEval recordings.')
    plan = dict(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL,
                tokenizer_revision=TOKENIZER_REVISION, task='ifeval', bank=bank, records=records, seed=42,
                official_revision=official.REVISION, prompt=official.PROMPT, params=dict(kimi.PARAMS, max_new_tokens=1024))
    freeze(OUTPUT/'run_config.json', plan)
    reuse_previous(plan)
    print(f"Verified 508 neutral/angry/happy triplets. Written wrapper: {plan['prompt']}", flush=True)
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
        comparisons[mode] = {}
        for ref, style in (('neutral','angry'), ('neutral','happy'), ('angry','happy')):
            result = paired([(scored[q,ref][mode]['prompt'],scored[q,style][mode]['prompt']) for q in ids])
            comparisons[mode][f'{style}_vs_{ref}'] = result
            n = len(ids)
            a, b = result['baseline_correct'], result['steered_correct']
            print(f"{mode}_prompt: {ref} {a}/{n} ({100*a/n:.1f}%) -> {style} {b}/{n} ({100*b/n:.1f}%); "
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
    p.add_argument('--source', type=Path, default=os.getenv('IFEVAL_AUDIO_DIR'))
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
