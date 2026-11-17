"""Describe saved response lengths within matched correctness outcome groups. CPU only."""
import argparse
import json
from pathlib import Path
import statistics

from steer import read_rows


def describe(path, reference, tests, metric='flexible-numeric', examples=0, compact=False):
    rows = read_rows(path)
    groups = {}
    for r in rows:
        condition = r.get('condition', r.get('style'))
        if condition not in [reference, *tests]: continue
        if metric not in r.get('scores', {}):
            raise ValueError(f'Missing {metric} scores; run the experiment summary first.')
        if not isinstance(r['scores'][metric], bool): raise ValueError('Expected boolean correctness score.')
        tokens = r.get('response_tokens')
        if not isinstance(tokens, int) or tokens < 0: raise ValueError('Missing/invalid saved token count.')
        key = r['question_id'], r.get('voice', r.get('speaker'))
        if key in groups.setdefault(condition, {}): raise ValueError('Duplicate condition/question/voice.')
        groups[condition][key] = r
    if reference not in groups: raise ValueError(f'No {reference} answers.')
    for test in tests:
        if test not in groups:
            print(f'{path}: no {test} answers.'); continue
        a, b = groups[reference], groups[test]
        keys = sorted(set(a)&set(b))
        if compact:
            print(f'\n{test} vs {reference}: {len(keys)} matched questions')
            print(f'Outcome                          N  {reference} mean/median  {test} mean/median')
        else:
            print(f'\n{path}\n{test} versus {reference}; {len(keys)} matched pairs; correctness={metric}.')
            print('Outcome                          N   Ref mean/median   Test mean/median   Paired delta mean/median  Shorter/equal/longer')
        for name, ref_ok, test_ok in [('Test wins / reference loses', False, True),
                                     ('Reference wins / test loses', True, False),
                                     ('Both correct', True, True), ('Both incorrect', False, False)]:
            if compact and ref_ok and test_ok: continue
            selected = [k for k in keys if a[k]['scores'][metric] == ref_ok and b[k]['scores'][metric] == test_ok]
            if not selected:
                print(f'{name:31} 0'); continue
            aa = [a[k]['response_tokens'] for k in selected]; bb = [b[k]['response_tokens'] for k in selected]
            delta = [y-x for x, y in zip(aa, bb)]
            def cell(x): return f'{statistics.mean(x):.1f}/{statistics.median(x):.1f}'
            signs = f'{sum(d<0 for d in delta)}/{sum(d==0 for d in delta)}/{sum(d>0 for d in delta)}'
            if compact:
                label = (f'{test} wins / {reference} loses' if test_ok else
                         f'{reference} wins / {test} loses' if ref_ok else 'Both lose')
                print(f'{label:31} {len(selected):3d} {cell(aa):>17} {cell(bb):>18}')
                continue
            print(f'{name:31} {len(selected):3d} {cell(aa):>17} {cell(bb):>18} {cell(delta):>26} {signs:>21}')
            if ref_ok == test_ok or examples <= 0: continue
            # Evenly spaced examples by paired length difference, not selected for a preferred narrative.
            ordered = sorted(selected, key=lambda k: (b[k]['response_tokens']-a[k]['response_tokens'], k))
            indices = sorted({round((i+.5)*len(ordered)/min(examples, len(ordered))-.5)
                              for i in range(min(examples, len(ordered)))})
            for index in indices:
                k = ordered[index]
                print(f'\nEXAMPLE {k[0]} / {name}')
                for label, item in ((reference, a[k]), (test, b[k])):
                    print(f'{label} ({item["response_tokens"]} tokens):\n{item["response"]}\n')
    if not compact:
        print('\nLengths include native generated suffix tokens (including EOS where saved). '
              'These outcome-selected descriptions do not establish that length causes correctness or measure internal reasoning.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path', type=Path, required=True, help='Latest rescored.jsonl from experiment summary.')
    p.add_argument('--reference', default='neutral')
    p.add_argument('--tests', nargs='+', default=['angry'])
    p.add_argument('--metric', default='flexible-numeric')
    p.add_argument('--examples', type=int, default=0)
    p.add_argument('--compact', action='store_true', help='Only test wins, reference wins and both lose; mean/median tokens.')
    args = p.parse_args()
    if args.examples < 0: p.error('--examples must be nonnegative')
    describe(args.path, args.reference, args.tests, args.metric, args.examples, args.compact)


if __name__ == '__main__': main()
