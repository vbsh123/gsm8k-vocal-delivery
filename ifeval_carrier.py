"""Evaluate unchanged written IFEval prompts with a neutral or angry task announcement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import sys
import wave
from hashlib import sha256

import run_diffa_spoken_prompt as pilot
import run_spoken_ifeval as official
from run_reasoning_transfer import bank_file, source_records
from rescore_audio_steering import paired
from run_arithmetic_audio_pilot import accuracy_interaction

OUTPUT = Path('outputs/diffa-ifeval-written-carrier')
SPOKEN = Path('outputs/emotion-block-night/diffa/ifeval')
PHRASE = 'Here is the task.'
RATES = (-10, 0, 10)
STYLES = ('neutral', 'angry')
FIELDS = ('key', 'prompt', 'instruction_id_list', 'kwargs')
winners = pilot.winners
night = pilot.evaluation.night


def carrier_clips(folder):
    bank = {'carrier': dict(question_id='carrier', question=PHRASE, answer='')}
    clips = {}
    waveforms = set()
    for rate in RATES:
        source = folder/f'rate{rate:+d}'
        source.mkdir(parents=True, exist_ok=True)
        questions = source/'questions.jsonl'
        bank_file(questions, list(bank.values()))
        subprocess.run([sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
                        '--input-jsonl', str(questions), '--azure-rate-offset', str(rate),
                        '--styles', *STYLES, '--output', str(source), '--resume', '--retry-passes', '5'], check=True)
        records = source_records(source, bank, STYLES)
        for style in STYLES:
            clip = records['carrier', style]
            with wave.open(clip['audio_file']) as wav:
                samples = wav.readframes(wav.getnframes())
            digest = sha256(samples).hexdigest()
            if digest in waveforms:
                raise ValueError(f'Duplicate carrier waveform at {rate:+d}% / {style}: {clip["audio_file"]}')
            waveforms.add(digest)
            clips[rate, style] = clip
    return clips


def build_plan(bank, clips):
    rng = random.Random(42)
    order = sorted(bank)
    rng.shuffle(order)
    strata = defaultdict(list)
    for q in order:
        strata[len(bank[q]['instruction_id_list'])].append(q)
    rates = {}
    for ids in strata.values():
        rotation = rng.randrange(len(RATES))
        for i, q in enumerate(ids):
            rates[q] = RATES[(rotation+i) % len(RATES)]
    records = []
    for q in order:
        for style in STYLES:
            clip = clips[rates[q], style]
            records.append(dict(question_id=q, condition=style, style=style, voice=clip['voice'],
                                audio_file=clip['audio_file'], audio_sha256=clip['audio_sha256'],
                                carrier_rate=rates[q], spoken_prompt=PHRASE, prompt=bank[q]['prompt']))
    recipe = dict(night.vanilla.RECIPE, max_new_tokens=1024)
    return dict(protocol='diffa-ifeval-written-carrier-v1', task='ifeval', name='ifeval', model='diffa',
                bank=bank, records=records, recipe=recipe, max_new_tokens=1024,
                seed=42, revision=official.REVISION, primary_metric='strict_prompt')


def report(folder, plan, parent):
    saved = pilot.saved(folder, plan)
    ids = [q for q in plan['bank'] if all((q,s) in saved for s in STYLES)]
    print(f'Written IFEval: {len(saved)}/{len(plan["records"])} answers, {len(ids)} matched prompts.')
    if not ids:
        return
    rows, receipt = winners.score_rows(dict(target=plan), [saved[q,s] for q in ids for s in STYLES], folder)
    groups = {s:{r['question_id']:r for r in rows if r['condition']==s} for s in STYLES}
    lengths, comparisons = {}, {}
    for s in STYLES:
        tokens = [groups[s][q]['response_tokens'] for q in ids]
        lengths[s] = dict(mean=mean(tokens), median=median(tokens))
        print(f'{s}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    subsets = {'all': ids,
               'one instruction': [q for q in ids if len(plan['bank'][q]['instruction_id_list'])==1],
               'multiple instructions': [q for q in ids if len(plan['bank'][q]['instruction_id_list'])>1]}
    subsets.update({f'carrier {rate:+d}%':[q for q in ids if saved[q,'neutral']['carrier_rate']==rate] for rate in RATES})
    for label, selected in subsets.items():
        if not selected:
            continue
        comparisons[label] = {}
        for mode in ('strict', 'loose'):
            key = mode+'_prompt'
            result = paired([(groups['neutral'][q]['metrics'][key], groups['angry'][q]['metrics'][key]) for q in selected])
            comparisons[label][key] = result
            print(f'{label} / {key}: {result["baseline_correct"]}/{len(selected)} -> '
                  f'{result["steered_correct"]}/{len(selected)}, {result["gain_pp"]:+.2f} pp, '
                  f'95% CI {result["ci95_gain_pp"]}, p={result["mcnemar_exact_p"]:.4g}')
    instructions = {}
    for mode in ('strict', 'loose'):
        totals = {}
        for s in STYLES:
            bits = [b for q in ids for b in groups[s][q]['official_scores'][mode]['instructions']]
            totals[s] = dict(passed=sum(bits), total=len(bits))
        instructions[mode] = totals
        print(f'{mode} individual constraints: neutral {totals["neutral"]["passed"]}/{len(bits)}, '
              f'angry {totals["angry"]["passed"]}/{len(bits)}')
    change = {}
    if (parent/'run_config.json').is_file():
        old = json.loads((parent/'run_config.json').read_text())
        if (old['model'] != 'diffa' or old['task'] != 'ifeval' or old['prompt'] != official.PROMPT
                or old['recipe'] != plan['recipe']):
            raise ValueError(f'Incompatible spoken-instruction reference at {parent}.')
        earlier = night.raw_rows(parent, old)
        common = [q for q in ids if all((q,s) in earlier for s in STYLES)]
        if any(any(old['bank'][q][k] != plan['bank'][q][k] for k in FIELDS) for q in common):
            raise ValueError('The two placements have different benchmark prompts or scoring metadata.')
        if common:
            scored, _ = winners.score_rows(dict(target=plan), [earlier[q,s] for q in common for s in STYLES], folder)
            joint = {s:{r['question_id']:r for r in scored if r['condition']==s} for s in STYLES}
            joint.update({s+'-reasoning':groups[s] for s in STYLES})
            for metric in ('strict_prompt', 'loose_prompt'):
                value = accuracy_interaction(joint, metric)
                change[metric] = value
                print(f'Written minus spoken change in angry-neutral gap / {metric}, N={len(common)}: '
                      f'{value["interaction_pp"]:+.2f} pp, CI={value["ci95_pp"]}, p={value["p"]}')
    if not change:
        print('Spoken-instruction comparison awaits saved neutral/angry pairs on this machine.')
    winners.atomic_text(folder/'official-scored.jsonl', ''.join(json.dumps(r)+'\n' for r in rows))
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), lengths=lengths, comparisons=comparisons,
                        instruction_totals=instructions, written_minus_spoken=change, scorer=receipt), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'summary'), default='run')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--spoken-parent', type=Path, default=SPOKEN)
    args = parser.parse_args()
    folder, parent = args.output, args.spoken_parent
    config = folder/'run_config.json'
    if args.mode == 'summary':
        if config.is_file():
            report(folder, json.loads(config.read_text()), parent)
        else:
            print('Written IFEval: awaiting preparation.')
        return
    with pilot.job_lock(folder):
        package = official.setup_official()
        examples = pilot.read_rows(package/'data/input_data.jsonl')
        bank = {f'ifeval_{r["key"]}':{k:r[k] for k in FIELDS} for r in examples}
        if len(bank) != 541:
            raise ValueError('Expected 541 distinct official prompts.')
        if (parent/'run_config.json').is_file():
            old = json.loads((parent/'run_config.json').read_text())['bank']
            if set(old)!=set(bank) or any(any(old[q][k]!=bank[q][k] for k in FIELDS) for q in bank):
                raise ValueError('Spoken run differs from the official benchmark IDs or scoring inputs.')
        plan = build_plan(bank, carrier_clips(folder/'audio'))
        winners.freeze(config, plan)
        print('541 written IFEval prompts, two deliveries each. Spoken words: '+PHRASE, flush=True)
        official.score(examples[0], 'Preflight check.', official.official_scorer())
        if args.mode == 'prepare':
            return
        if len(pilot.saved(folder, plan, repair=True)) < len(plan['records']):
            with pilot.gpu_queue():
                from diffa_model_receipts import model_revisions
                from src.dataloader import build_chat_prompt
                winners.freeze(folder/'model-revisions.json', model_revisions())
                pilot.generate(folder, plan, night.DiffaBackend(), build_chat_prompt)
        report(folder, plan, parent)


if __name__ == '__main__':
    main()
