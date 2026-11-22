"""Compact saved summaries for original DIFFA happy, shared-three DIFFA and fresh Phi ZIP checks."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

TASKS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')


def primary(task):
    return 'strict_prompt' if task == 'ifeval' else 'flexible-extract' if task.startswith('bbh') else 'flexible-numeric'


def collect(root):
    result = []
    for task in TASKS:
        paths = [('DIFFA original happy', root/'diffa-happy-winners'/task/'strength-1/summary.json')]
        base = root/'diffa-gsm-shared-three-zeroshot-hashes' if task == 'gsm8k' else root/'diffa-shared-three-tasks'/task
        paths += [(f'DIFFA shared {e}', base/e/'primary/summary.json') for e in ('angry', 'happy')]
        paths += [(f'Phi ZIP {target} -1', root/'phi4-diffa-cohorts-three-directions'/task/f'summary-{target}.json')
                  for target in ('L0-audio','L11-answer','L30-answer')]
        for workflow, path in paths:
            if not path.exists():
                result.append(dict(workflow=workflow, dataset=task, status='No saved summary'))
                continue
            summary = json.loads(path.read_text())
            paired = summary.get('paired', [])
            if workflow.startswith('Phi'):
                paired = [r for r in paired if r.get('split') == 'held-out' and r.get('condition') == 'steered']
            else:
                paired = [r for r in paired if r.get('strength') == 1.]
            row = dict(workflow=workflow, dataset=task, status='Available', primary_metric=primary(task), summary_path=str(path))
            config = path.parent/'run_config.json'
            if config.exists():
                row['run_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
            for metric, label in [(primary(task), 'primary'), ('strict_prompt' if task == 'ifeval' else 'strict-match', 'strict'),
                                  ('loose_prompt' if task == 'ifeval' else 'flexible-extract', 'flexible')]:
                value = next((r for r in paired if r['metric'] == metric), None)
                if value:
                    row.update({label+'_gain_pp': value['gain_pp'], label+'_p': value['mcnemar_exact_p'],
                                label+'_ci': value['ci95_gain_pp'], 'pairs': value['pairs']})
            if 'primary_gain_pp' not in row:
                row['status'] = 'No matched steering results yet'
            if 'verification' in summary and summary['verification'].get('historical_comparison', True):
                verification = summary['verification']
                row.update(baseline_checked=verification['matched'], baseline_changed=len(verification['differences']))
            result.append(row)
    return result


def merge(records):
    chosen = {}
    for row in records:
        key = row['workflow'], row['dataset']
        old = chosen.get(key)
        if old is None:
            chosen[key] = row
            continue
        old_rank = ('primary_gain_pp' in old, old.get('pairs', 0))
        rank = ('primary_gain_pp' in row, row.get('pairs', 0))
        if old_rank[0] and rank[0] and old.get('run_config_sha256') and row.get('run_config_sha256') and old['run_config_sha256'] != row['run_config_sha256']:
            raise ValueError(f'Different frozen runs cannot be merged under one label: {key}')
        if old_rank == rank and rank[0] and any(old.get(k) != row.get(k) for k in ('primary_gain_pp','primary_p','strict_gain_pp','strict_p')):
            raise ValueError(f'Conflicting same-size summaries: {key}')
        if rank > old_rank:
            chosen[key] = row
    return list(chosen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('outputs/three-machine-report'))
    parser.add_argument('--inputs', type=Path, nargs='+', help='Merge downloaded summary.json files from the three machines.')
    args = parser.parse_args()
    records = merge([row for path in args.inputs for row in json.loads(path.read_text())]) if args.inputs else collect(args.root)
    print(f'{"Workflow":24} {"Dataset":11} {"N":>5} {"Primary gain/p":>20} {"Strict gain/p":>20}')
    for row in records:
        def value(label):
            return f'{row[label+"_gain_pp"]:+.2f}/{row[label+"_p"]:.4g}' if label+'_gain_pp' in row else '—'
        print(f'{row["workflow"]:24} {row["dataset"]:11} {row.get("pairs",0):5} {value("primary"):>20} {value("strict"):>20}')
        if row['status'] != 'Available':
            print(' ', row['status'])
        if 'baseline_checked' in row:
            print(f'  Historical baseline check: {row["baseline_checked"]} compared, {row["baseline_changed"]} changed.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
    fields = sorted({key for row in records for key in row})
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print('Primary: numeric equivalence for math, flexible extraction for BBH, strict prompt compliance for IFEval.')
    print('Local saved summaries only; missing remote runs are not interpreted as failed experiments. Unadjusted tests.')
    print('Report files:', args.output)


if __name__ == '__main__':
    main()
