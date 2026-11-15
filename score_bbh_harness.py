"""CPU-only BBH saved-answer scoring through installed EleutherAI task APIs.

Audio-adapted generations, not the official text/few-shot inference protocol.
Original response strings are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path


from rescore_audio_steering import HarnessTaskScorer, paired
from steer import read_rows

STYLES = ('neutral', 'angry', 'sad', 'happy')


class HarnessScorer(HarnessTaskScorer):
    def __init__(self, cache, objects='three'):
        if objects not in ('three', 'five', 'seven'):
            raise ValueError('BBH logical deduction supports three, five or seven objects')
        super().__init__(f'bbh_cot_zeroshot_logical_deduction_{objects}_objects', cache)


def report_saved(output, source, destination=None, cache=None):
    from run_bbh_logical200 import SOURCE_SHA256
    output, source = Path(output), Path(source)
    destination = Path(destination) if destination else output/'harness'
    original = (source/'logical_deduction_three_objects-source.json').read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official BBH source checksum differs')
    official = json.loads(original)['examples']
    bank = {q['question_id']: q for q in read_rows(source/'questions.jsonl')}
    for q in bank.values():
        expected = official[q['source_index']]
        if (q['original_input'] != expected['input'] or q['original_target'] != expected['target']
                or q['answer'] != expected['target'][1]):
            raise ValueError('Selected question differs from official BBH source')
    latest = {}
    for row in read_rows(output/'results.jsonl'):
        condition = row['condition']
        if condition not in STYLES:
            continue
        key = row['question_id'], row['voice'], condition
        if row['reference_answer'] != bank[key[0]]['answer']:
            raise ValueError('Saved reference differs from original BBH target')
        if key in latest and latest[key]['response'] != row['response']:
            raise ValueError('Conflicting saved responses for the same question/voice/style')
        latest[key] = row
    groups = {style:{k[:2] for k in latest if k[2] == style} for style in STYLES}
    matched = sorted(set.intersection(*groups.values()))
    keys = [(*key, style) for key in matched for style in STYLES]
    rows = [latest[key] for key in keys]
    destination.mkdir(parents=True, exist_ok=True)
    scorer = HarnessScorer(cache or destination/'upstream')
    docs = [dict(input=bank[r['question_id']]['original_input'],
                 target=bank[r['question_id']]['original_target']) for r in rows]
    scores = scorer.score([r['response'] for r in rows], docs)
    rescored = [dict(row, harness={name: values[i] for name,values in scores.items()}) for i,row in enumerate(rows)]
    accuracy, comparisons = [], []
    print('\nBBH: upstream EleutherAI harness CoT zero-shot filters + exact_match metric.')
    print('Audio-adapted saved generations; not a full official text evaluation. Responses are scored verbatim.')
    print(f'{len(matched)} questions/voices completed in ALL FOUR deliveries; no outcome/classifier filtering.')
    for metric in scores:
        by_style = {style:{(r['question_id'],r['voice']): r['harness'][metric]['correct']
                           for r in rescored if r['condition'] == style} for style in STYLES}
        for style,values in by_style.items():
            n = len(values); correct = sum(values.values())
            print(f'{metric:18s} {style:8s} {correct}/{n}' + (f' ({correct/n:.1%})' if n else ''))
            accuracy.append(dict(metric=metric, condition=style, n=n, correct=correct))
        if matched:
            for style in STYLES[1:]:
                result = paired([(by_style['neutral'][k], by_style[style][k]) for k in matched])
                comparisons.append(dict(metric=metric, condition=style, **result))
                print(f'  {style} vs neutral: {result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}')
    summary = dict(scorer=scorer.provenance, matched_questions=len(matched),
                   unmatched_rows_excluded=len(latest)-len(rows), accuracy=accuracy, paired=comparisons,
                   inference_protocol='Audio-adapted saved generations; raw responses and recorded generation settings unchanged by this scoring audit')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (destination/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rescored))
    if any('Final answer:' in r.get('prompt', '') for r in rows):
        print('Legacy Final-answer prompt detected: strict extraction expects a different upstream answer phrase.')
    else:
        print('Strict extraction expects the upstream answer phrase, e.g. The answer is (A).')
    print('Flexible extraction may match choices mentioned in reasoning; upstream behavior is preserved. Paired tests are unadjusted.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    parser.add_argument('--output', type=Path, default=Path('outputs/bbh-logical200-diffa2'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--upstream-cache', type=Path)
    args = parser.parse_args()
    report_saved(args.output, args.source, args.destination, args.upstream_cache)


if __name__ == '__main__':
    main()
