"""Swap only Ready and reasoning on the original 270 neutral MultiArith recordings."""
import json
from pathlib import Path
import re
from statistics import mean, median

from multiarith_instruction_checks import THIRD_ASK
from rescore_audio_steering import paired
from steer import read_rows
import winning_diffa_sweeps as winners


def prepare(baseline, folder, comparison, reference):
    source = baseline / 'run_config.json'
    original = json.loads(source.read_text())
    archive = json.loads(comparison.read_text())
    prompt = original['prompt']
    reasoning = 'Give concise step-by-step reasoning with only the necessary calculations.'
    adjacent = reasoning + ' ' + THIRD_ASK
    if prompt != archive['prompt'] or prompt.count(adjacent) != 1:
        raise ValueError('Expected the archived original three-ask wording.')
    ids = list(json.loads(reference.read_text())['questions'])[20:]
    if len(ids) != 270 or ids != original['question_order'][20:]:
        raise ValueError('Expected the same 270 questions as the sequenced-rest run.')
    selected = set(ids)
    records = [r for r in original['records'] if r['condition'] == 'neutral' and r['question_id'] in selected]
    if [r['question_id'] for r in records] != ids:
        raise ValueError('Expected exactly one original neutral recording per question.')
    source_hash = winners.digest(source)
    expected = {r['question_id']: r for r in records}
    original_rows = {}
    for row in read_rows(baseline / 'rescored.jsonl'):
        q = row['question_id']
        if q not in selected or row['condition'] != 'neutral':
            continue
        if (q in original_rows or row['run_config_sha256'] != source_hash or row['prompt'] != prompt
                or row['reference_answer'] != expected[q]['reference_answer']
                or row['audio_sha256'] != expected[q]['audio_sha256']):
            raise ValueError(f'Original score or recording mismatch on {q}')
        original_rows[q] = row
    if original_rows.keys() != selected:
        raise ValueError('Missing original neutral scores in the three-ask rescored.jsonl.')
    plan = dict(original, original_prompt=prompt, prompt=prompt.replace(adjacent, THIRD_ASK+' '+reasoning),
                variant='neutral-order-only', question_order=ids, records=records,
                bank={q: original['bank'][q] for q in ids}, primary_metric='flexible-numeric',
                source_config_sha256=source_hash)
    winners.freeze(folder/'run_config.json', plan)
    return plan, original_rows


def ready_only(row):
    return bool(re.fullmatch(r'\s*Ready[.!?]?\s*', row['response'], re.I))


def report(folder, plan, original, saved):
    ids = [q for q in plan['question_order'] if (q, 'neutral') in saved]
    print(f'MultiArith order only: {len(ids)}/270 matched neutral answers.')
    if not ids:
        return
    scored, receipt = winners.score_rows(dict(target=plan), [saved[q,'neutral'] for q in ids], folder/'scoring')
    matched = [(original[r['question_id']], r) for r in scored]
    outcomes = {}
    for metric in ('ready-only', 'flexible-numeric'):
        values = [(ready_only(a), ready_only(b)) if metric == 'ready-only' else
                  (a['metrics'][metric], b['metrics'][metric]) for a,b in matched]
        s = paired(values)
        outcomes[metric] = s
        print(f'{metric}: original {s["baseline_correct"]}/{len(ids)} -> swapped {s["steered_correct"]}/{len(ids)}; '
              f'{s["gain_pp"]:+.2f} pp; 95% CI {s["ci95_gain_pp"]}; p={s["mcnemar_exact_p"]:.4g}')
    for label, rows in [('original', [a for a,_ in matched]), ('swapped', scored)]:
        lengths = [r['response_tokens'] for r in rows]
        print(f'{label} mean/median tokens: {mean(lengths):.1f}/{median(lengths):g}')
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), comparisons=outcomes, scorer=receipt), indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
