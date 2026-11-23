"""Print saved per-layer emotion/neutral activation changes. Standard library only."""
import argparse
import csv
import math
from pathlib import Path


def read_table(path, group):
    if not path.exists():
        return {}
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['group'] == group]
    result = {}
    for row in rows:
        layer = int(row['layer'])
        if layer in result:
            raise ValueError(f'Duplicate {group}/L{layer} in {path}')
        result[layer] = row
    return result


def value(row, field, digits=2):
    if row is None:
        return '-'
    number = float(row[field])
    return f'{number:.{digits}f}' if math.isfinite(number) else 'n/a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    parser.add_argument('--emotions', choices=('angry', 'happy', 'sad'), nargs='+', default=['angry'])
    parser.add_argument('--regimes', choices=('random', 'agreed'), nargs='+', default=['random', 'agreed'])
    parser.add_argument('--group', choices=('answer', 'acoustic', 'semantic', 'combined', 'all'), default='answer')
    parser.add_argument('--cosine', action='store_true', help='Also show mean per-question neutral/emotion cosine.')
    parser.add_argument('--median', action='store_true', help='Also show median change%%.')
    args = parser.parse_args()
    print('Change% = mean across questions of 100 * ||emotion-neutral|| / ||neutral||.')
    print('States are token/forward-pooled means. Layers are zero-based. No inference or reasoning-direction comparison.')
    groups = ['answer', 'acoustic', 'semantic', 'combined'] if args.group == 'all' else [args.group]
    for emotion in dict.fromkeys(args.emotions):
        for group in groups:
            tables = {r: read_table(args.root/emotion/r/'layer-drift.csv', group) for r in dict.fromkeys(args.regimes)}
            layers = sorted(set().union(*(set(t) for t in tables.values())))
            print(f'\n{emotion} | {group}')
            for regime, table in tables.items():
                if not table:
                    print(f'{regime}: no completed layer-drift.csv at {args.root/emotion/regime}; collection may still be running.')
                else:
                    counts = sorted({int(r['pairs']) for r in table.values()})
                    print(f'{regime}: {", ".join(map(str, counts))} collected pairs; {len(table)} layers.')
            if not layers:
                continue
            columns = [('mean_change_pct', 'mean%', 2)]
            if args.median:
                columns.append(('median_change_pct', 'median%', 2))
            if args.cosine:
                columns.append(('mean_cosine', 'cosine', 4))
            print(f'{"Layer":>5}'+''.join(f'{regime+" "+label:>17}' for regime in tables for _, label, _ in columns))
            for layer in layers:
                print(f'{layer:5d}'+''.join(f'{value(table.get(layer), field, digits):>17}'
                    for table in tables.values() for field, _, digits in columns))


if __name__ == '__main__':
    main()
