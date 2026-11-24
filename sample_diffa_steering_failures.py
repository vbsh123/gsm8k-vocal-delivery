"""Sample saved steering failures with paired neutral responses. CPU only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

def cases(config, rows, metric='flexible-numeric', kind='losses', strength=1.):
    if not math.isfinite(strength) or strength == 0.:
        raise ValueError('Select a finite, nonzero steering strength.')
    expected = {(r['question_id'], r['voice']): r for r in config['target']['records']}
    grouped = {0.: {}, strength: {}}
    for row in rows:
        row_strength = row.get('strength')
        if row_strength not in grouped:
            continue
        key = row['question_id'], row['voice']
        record = expected.get(key)
        if (record is None or key in grouped[row_strength] or row.get('audio_sha256') != record['audio_sha256']
                or row.get('prompt') != config['target']['prompt']
                or (row_strength != 0. and row.get('vector_sha256') != config['source']['vector_sha256'])):
            raise ValueError('Scored snapshot differs from its frozen question/audio/prompt/vector configuration.')
        if metric not in row['metrics']:
            raise ValueError(f'Metric {metric!r} absent from saved scores.')
        grouped[row_strength][key] = row
    common = sorted(grouped[0.].keys() & grouped[strength].keys())
    selected = []
    for key in common:
        baseline, steered = grouped[0.][key], grouped[strength][key]
        before, after = bool(baseline['metrics'][metric]), bool(steered['metrics'][metric])
        if kind == 'all':
            include = True
        elif kind == 'losses':
            include = before and not after
        elif kind == 'errors':
            include = not after
        else:
            if 'format-compliant' not in baseline['metrics'] or 'format-compliant' not in steered['metrics']:
                raise ValueError('This run has no saved format-compliant metric.')
            include = baseline['metrics']['format-compliant'] and not steered['metrics']['format-compliant']
        if not include:
            continue
        example = config['target']['bank'][key[0]]
        selected.append(dict(question_id=key[0], voice=key[1], question=example['question'],
            gold=baseline.get('reference_answer', example.get('answer')), metric=metric, strength=strength,
            baseline_correct=before, steered_correct=after,
            baseline_response=baseline['response'], steered_response=steered['response'],
            baseline_predictions=baseline.get('predictions', {}), steered_predictions=steered.get('predictions', {}),
            baseline_metrics=baseline['metrics'], steered_metrics=steered['metrics'],
            baseline_tokens=baseline.get('response_tokens'), steered_tokens=steered.get('response_tokens'),
            baseline_audio_file=baseline.get('audio_file', expected[key].get('audio_file')),
            steered_audio_file=steered.get('audio_file', expected[key].get('audio_file'))))
    return selected, len(common)


def lengths(items):
    tokenizer = None
    for item in items:
        for prefix in ('baseline', 'steered'):
            if item[prefix+'_tokens'] is None:
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', local_files_only=True)
                item[prefix+'_tokens'] = len(tokenizer.encode(item[prefix+'_response'], add_special_tokens=False))
        item['extra_tokens'] = item['steered_tokens']-item['baseline_tokens']


def render(items, heading):
    blocks = [heading]
    for i, item in enumerate(items, 1):
        blocks.append(f'\nCASE {i}: {item["question_id"]}\nQUESTION: {item["question"]}\nGOLD: {item["gold"]}\n'
            f'NEUTRAL: correct={item["baseline_correct"]}; tokens={item["baseline_tokens"]}; extracted={item["baseline_predictions"]}\n'
            f'AUDIO: {item.get("baseline_audio_file")}\n{item["baseline_response"]}\n\n'
            f'STEERED x{item.get("strength",1.):+g}: correct={item["steered_correct"]}; tokens={item["steered_tokens"]}; extra tokens={item["extra_tokens"]:+d}; '
            f'extracted={item["steered_predictions"]}\nAUDIO: {item.get("steered_audio_file")}\n{item["steered_response"]}\n')
    return '\n'.join(blocks)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emotion', choices=('angry', 'happy', 'sad'), default='angry')
    parser.add_argument('--regime', choices=('random', 'agreed'), default='random')
    parser.add_argument('--root', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    parser.add_argument('--run', type=Path, help='Override the steering folder for another dataset/run.')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--strength', type=float, default=1., help='Steered strength paired against x0, e.g. 2.')
    parser.add_argument('--longest', action='store_true', help='Show the longest steered responses rather than a random sample.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kind', choices=('losses', 'errors', 'format', 'all'), default='losses',
                        help='losses: neutral correct, steering wrong; errors: any steered error; format: new format failure; all: no accuracy filtering.')
    parser.add_argument('--metric', default='flexible-numeric')
    parser.add_argument('--min-extra-tokens', type=int, help='Only sample cases with at least this token increase.')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive.')
    if not math.isfinite(args.strength) or args.strength == 0.:
        parser.error('--strength must be finite and nonzero.')
    folder = args.run or args.root/args.emotion/args.regime/'strength-1-unfitted'
    if not (folder/'run_config.json').exists() or not (folder/'rescored.jsonl').exists():
        parser.error(f'No scored steering run at {folder}. Run the normal --mode summary first.')
    config_bytes = (folder/'run_config.json').read_bytes()
    score_bytes = (folder/'rescored.jsonl').read_bytes()
    config = json.loads(config_bytes)
    rows = [json.loads(line) for line in score_bytes.splitlines() if line.strip()]
    eligible, paired_count = cases(config, rows, args.metric, args.kind, args.strength)
    # Shuffle before computing lengths so ordinary samples only tokenize selected cases.
    random.Random(args.seed).shuffle(eligible)
    if args.min_extra_tokens is not None:
        lengths(eligible)
        eligible = [r for r in eligible if r['extra_tokens'] >= args.min_extra_tokens]
    if args.longest:
        lengths(eligible)
        eligible.sort(key=lambda r:(-r['steered_tokens'], r['question_id'], r['voice']))
    sample = eligible[:args.count]
    lengths(sample)
    regime = f'; regime={config["regime"]}' if config.get('regime') else ''
    heading = (f'Run: {folder}\nEmotion: {config.get("emotion", args.emotion)}{regime}; '
               f'layer={config["layer"]}, group={config["group"]}, strength={args.strength:+g}\n'
               f'Paired questions scored: {paired_count}; eligible {args.kind}: {len(eligible)}; '
               f'sampled: {len(sample)}; metric={args.metric}; seed={args.seed}\n'
               + ('Selected by longest steered output; diagnostic examples, not a representative sample.\n' if args.longest else '')
               + 'Saved responses and benchmark scores; no generation or causal error labeling.')
    report = render(sample, heading)
    print(report)
    destination = folder/'failure-samples'
    destination.mkdir(exist_ok=True)
    stem = f'{args.kind}-{args.metric}-seed{args.seed}-n{args.count}'
    if args.strength != 1.: stem += f'-a{args.strength:g}'
    if args.longest: stem += '-longest'
    if args.min_extra_tokens is not None:
        stem += f'-extra{args.min_extra_tokens}'
    (destination/(stem+'.txt')).write_text(report)
    (destination/(stem+'.json')).write_text(json.dumps(dict(run=str(folder), metric=args.metric, kind=args.kind,
        seed=args.seed, strength=args.strength, longest=args.longest,
        paired_questions=paired_count, eligible_cases=len(eligible), min_extra_tokens=args.min_extra_tokens,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(), scored_snapshot_sha256=hashlib.sha256(score_bytes).hexdigest(),
        vector=config['source'].get('vector'), vector_sha256=config['source']['vector_sha256'], cases=sample), indent=2)+'\n')
    print('Saved:', destination/(stem+'.txt'))


if __name__ == '__main__':
    main()
