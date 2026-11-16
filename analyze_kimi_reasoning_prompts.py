"""CPU comparison of original versus explicit-reasoning Kimi BBH prompts.

Accuracy uses the installed upstream task API. Word counts and answer-only
counts are descriptive, not automatic judgments of reasoning quality.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from steer import read_rows

REASONING_SENTENCE = 'Give concise step-by-step reasoning with only the necessary deductions.'
ANSWER_ONLY = re.compile(r'\s*The answer is \([ABC]\)\.\s*')


def aligned_rows(original, explicit):
    old = json.loads((original/'run_config.json').read_text())
    new = json.loads((explicit/'run_config.json').read_text())
    if old['prompt'].count("Let's think step by step.") != 1:
        raise ValueError('Original prompt is not the expected BBH reasoning prompt.')
    if new['prompt'] != old['prompt'].replace("Let's think step by step.", REASONING_SENTENCE):
        raise ValueError('More than the reasoning sentence changed between prompts.')
    for field in ('bank', 'seed', 'max_new_tokens', 'recipe'):
        if old[field] != new[field]:
            raise ValueError(f'Prompt experiments differ in {field}.')
    def audio_map(plan):
        return {(r['question_id'], r['voice'], r['style']): r['audio_sha256'] for r in plan['recordings']}
    audio_maps = {'original': audio_map(old), 'explicit': audio_map(new)}
    if audio_maps['original'] != audio_maps['explicit']:
        raise ValueError('Prompt experiments use different audio recordings.')
    groups = {}
    for variant, folder, plan in [('original', original, old), ('explicit', explicit, new)]:
        for row in read_rows(folder/'results.jsonl'):
            if row['style'] not in ('neutral', 'angry'):
                continue
            if row['prompt'] != plan['prompt'] or row['seed'] != plan['seed'] or row['max_new_tokens'] != plan['max_new_tokens']:
                raise ValueError('Saved response settings differ from the frozen prompt configuration.')
            if row['model'] != plan['recipe']['model'] or row['revision'] != plan['recipe']['revision']:
                raise ValueError('Saved response checkpoint differs from the frozen prompt configuration.')
            key = row['question_id'], row['voice']
            cell = variant, row['style']
            group = groups.setdefault(key, {})
            if cell in group:
                raise ValueError(f'Duplicate saved response: {key}, {cell}')
            if row['audio_sha256'] != audio_maps[variant][(*key, row['style'])]:
                raise ValueError('Saved response audio hash differs from its plan.')
            groups[key][cell] = row
    cells = [('original', 'neutral'), ('original', 'angry'), ('explicit', 'neutral'), ('explicit', 'angry')]
    common = sorted(k for k, g in groups.items() if all(c in g for c in cells))
    rows = [dict(groups[key][cell], prompt_variant=cell[0]) for key in common for cell in cells]
    return old, common, rows


def interaction_interval(differences):
    """Question-paired bootstrap of the delivery-by-prompt interaction, no McNemar test."""
    import numpy as np
    values = np.asarray(differences, dtype=int)
    categories = np.arange(-2, 3)
    probabilities = np.array([(values == x).sum() for x in categories], dtype=float)/len(values)
    draws = np.random.default_rng(42).multinomial(len(values), probabilities, size=200_000)
    bootstrap = draws @ categories / len(values) * 100
    return dict(gain_pp=float(values.mean()*100),
                ci95_gain_pp=[float(x) for x in np.percentile(bootstrap, [2.5, 97.5])])


def analyze(original, explicit, output):
    from score_bbh_harness import HarnessScorer
    from evaluate_bbh_audio_models import strict_format_bits
    from rescore_audio_steering import paired
    plan, common, rows = aligned_rows(original, explicit)
    if not common:
        print('No questions completed in all four cells yet: original/explicit × neutral/angry.')
        return
    output.mkdir(parents=True, exist_ok=True)
    scorer = HarnessScorer(output/'harness/upstream')
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    formats = strict_format_bits(scorer, [r['response'] for r in rows], docs)
    scored = [dict(r, harness={m: v[i] for m, v in values.items()},
                   **{'strict-format-extractable': formats[i]}, words=len(r['response'].split()),
                   answer_line_only=bool(ANSWER_ONLY.fullmatch(r['response'])),
                   literal_placeholder='<letter>' in r['response']) for i, r in enumerate(rows)]
    cells = {(variant, style): [r for r in scored if (r['prompt_variant'], r['style']) == (variant, style)]
             for variant in ('original', 'explicit') for style in ('neutral', 'angry')}
    print(f'\n{len(common)} SAME questions in all four cells; same audio, model and decoding.')
    print('Prompt    Delivery    N  Mean words  Median  Answer-only  Placeholder  Flexible  Strict  Valid format')
    summaries = []
    for (variant, style), group in cells.items():
        n = len(group)
        lengths = [r['words'] for r in group]
        counts = {m: sum(r['harness'][m]['correct'] for r in group) for m in values}
        counts['strict-format-extractable'] = sum(r['strict-format-extractable'] for r in group)
        info = dict(prompt=variant, delivery=style, n=n, mean_words=statistics.mean(lengths),
                    median_words=statistics.median(lengths), answer_line_only=sum(r['answer_line_only'] for r in group),
                    literal_placeholder=sum(r['literal_placeholder'] for r in group), correct=counts)
        summaries.append(info)
        print(f'{variant:9} {style:8} {n:4d} {info["mean_words"]:11.1f} {info["median_words"]:7.1f}'
              f' {info["answer_line_only"]:12d} {info["literal_placeholder"]:12d}'
              f' {counts["flexible-extract"]/n:8.1%} {counts["strict-match"]/n:7.1%}'
              f' {counts["strict-format-extractable"]/n:12.1%}')
    comparisons = []
    interactions = []
    for metric in ('flexible-extract', 'strict-match', 'strict-format-extractable'):
        def bits(cell):
            return [r[metric] if metric == 'strict-format-extractable' else r['harness'][metric]['correct'] for r in cells[cell]]
        on, oa, en, ea = [bits(c) for c in [('original', 'neutral'), ('original', 'angry'), ('explicit', 'neutral'), ('explicit', 'angry')]]
        for label, baseline, test in [('original: angry vs neutral', on, oa), ('explicit: angry vs neutral', en, ea),
                                      ('neutral: explicit vs original', on, en), ('angry: explicit vs original', oa, ea)]:
            result = paired(list(zip(baseline, test)))
            comparisons.append(dict(metric=metric, comparison=label, **result))
            print(f'{metric} / {label}: {result["gain_pp"]:+.2f} pp; '
                  f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
        delta = [int(d)-int(c)-int(b)+int(a) for a, b, c, d in zip(on, oa, en, ea)]
        interaction = interaction_interval(delta)
        interactions.append(dict(metric=metric, **interaction))
        print(f'{metric} / change in angry-neutral gap under explicit reasoning: '
              f'{interaction["gain_pp"]:+.2f} pp; paired bootstrap CI={interaction["ci95_gain_pp"]}')
    # Blind review distinguishes actual intermediate deductions from verbosity.
    # Words, correctness, delivery and prompt labels are intentionally absent from the review sheet.
    review = list(scored)
    random.Random(42).shuffle(review)
    review_file = output/'reasoning-review.csv'
    previous_review = {}
    if review_file.exists():
        with review_file.open(newline='') as stream:
            previous_review = {r['sample_id']: r for r in csv.DictReader(stream)}
    def review_id(row):
        key = json.dumps([row['question_id'], row['voice'], row['prompt_variant'], row['style']])
        return 'R'+hashlib.sha256(key.encode()).hexdigest()[:16]
    if not previous_review.keys() <= {review_id(r) for r in review}:
        raise ValueError('Current cohort is missing previously reviewed responses. Existing annotations retained; use another output directory.')
    for r in review:
        previous = previous_review.get(review_id(r))
        if previous and (previous['response'] != r['response']
                         or previous['question'] != plan['bank'][r['question_id']]['original_input']):
            raise ValueError('A reviewed response changed. Existing annotations retained; use another output directory.')
    keys = []
    with review_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'question', 'response', 'reasoning_status', 'reasoning_correct'])
        writer.writeheader()
        for r in review:
            sample_id = review_id(r)
            previous = previous_review.get(sample_id, {})
            writer.writerow(dict(sample_id=sample_id, question=plan['bank'][r['question_id']]['original_input'],
                                 response=r['response'], reasoning_status=previous.get('reasoning_status', ''),
                                 reasoning_correct=previous.get('reasoning_correct', '')))
            keys.append(dict(sample_id=sample_id, question_id=r['question_id'], voice=r['voice'],
                             prompt_variant=r['prompt_variant'], style=r['style']))
    (output/'reasoning-review-key.json').write_text(json.dumps(keys, indent=2)+'\n')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(matched_questions=len(common),
        primary_metric='flexible-extract', primary_contrast='delivery-by-prompt interaction', cells=summaries,
        comparisons=comparisons, delivery_by_prompt_interactions=interactions, scorer=scorer.provenance,
        note='All comparisons exploratory/unadjusted; bootstrap units are questions. Word counts do not measure reasoning quality. This prompt intervention alone does not establish causal mediation. Original answers untouched.'), indent=2)+'\n')
    print('Saved blind reasoning-review.csv. Labels: reasoning_status = none/intermediate_steps/unclear; '
          'reasoning_correct = yes/no/unclear (leave blank when no intermediate reasoning).')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=Path('outputs/bbh-three-regenerated-models/kimi'))
    p.add_argument('--explicit', type=Path, default=Path('outputs/bbh-three-explicit-reasoning/kimi'))
    p.add_argument('--output', type=Path, default=Path('outputs/kimi-bbh-reasoning-comparison'))
    args = p.parse_args()
    analyze(args.original, args.explicit, args.output)


if __name__ == '__main__':
    main()
