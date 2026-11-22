"""Report saved emotion2vec labels on exact original happy collection clips; no inference/filtering."""
import argparse
from collections import Counter
import json
from pathlib import Path

MODEL = 'emotion2vec/emotion2vec_plus_large'
DATASETS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')


def read_labels(root):
    labels = {}
    for path in sorted(root.rglob('results-emotion2vec.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get('model') != MODEL or row.get('processing_mode') != 'whole_recording'
                    or row.get('classification_status') != 'classified'):
                continue
            key = row.get('audio_sha256') or row.get('clip_id')
            if not key:
                continue
            if key in labels and labels[key] != row['emotion']:
                raise ValueError(f'Conflicting saved e2v labels for {key}')
            labels[key] = row['emotion']
    return labels


def angry_records(value, found=None, hint=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if value.get('audio_sha256') and value.get('question_id'):
            style = value.get('style') or value.get('condition') or hint
            if style == 'angry':
                q = value['question_id']
                if q in found and found[q]['audio_sha256'] != value['audio_sha256']:
                    raise ValueError(f'Ambiguous original angry audio for {q}')
                found[q] = value
        else:
            for key, child in value.items():
                angry_records(child, found, key if key in ('angry', 'neutral', 'happy') else hint)
    elif isinstance(value, list):
        for child in value:
            angry_records(child, found, hint)
    return found


def counts(ids, records, style, labels):
    predictions = Counter(labels[r['audio_sha256']] for q in ids
                          if (r := records.get(q)) and r['audio_sha256'] in labels)
    classified = sum(predictions.values())
    available = sum(q in records for q in ids)
    return dict(total=len(ids), exact_records=available, classified=classified,
                agreed=predictions[style], pending=available-classified,
                missing_records=len(ids)-available,
                agreement_pct=100*predictions[style]/classified if classified else None,
                predicted_labels=dict(predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--labels-root', type=Path, default=Path('outputs'))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    labels = read_labels(args.labels_root)
    report = {}
    print('Exact original collection IDs/audio hashes. Saved whole-recording e2v labels only; no filtering.')
    print(f'{"Dataset":12} {"Style":8} {"Cohort":>7} {"Classified":>11} {"Agreed":>7} {"Match%":>8} {"Pending":>8} {"No record":>10}')
    for name in args.datasets:
        path = args.root/name/'run_config.json'
        if not path.exists():
            print(f'{name}: happy plan not prepared yet.')
            continue
        plan = json.loads(path.read_text())
        ids = plan['collection_ids']
        records = {style: {p['question_id']: p[key] for p in plan['pairs']}
                   for style, key in (('happy', 'emotion'), ('neutral', 'neutral'))}
        source = json.loads(Path(plan['target']['config_file']).read_text())
        records['angry'] = angry_records(source)
        if not set(ids) <= records['angry'].keys():
            # GSM's legacy vector has clip IDs rather than hashes. Use the saved
            # original-cohort audit, never a different recording with the same ID.
            audit = args.labels_root/'diffa-angry-agreed/audit/cohorts.json'
            if audit.exists():
                cohort = json.loads(audit.read_text()).get('datasets', {}).get(name, {})
                pinned = cohort.get('pinned', {})
                for pair in plan['pairs']:
                    q = pair['question_id']
                    neutral = pinned.get(q+'|neutral', {}).get('record', {})
                    angry = pinned.get(q+'|angry', {}).get('record')
                    if (angry and q not in records['angry'] and
                            neutral.get('audio_sha256') == pair['neutral']['audio_sha256']):
                        records['angry'][q] = angry
        report[name] = {}
        for style in ('neutral', 'angry', 'happy'):
            result = counts(ids, records[style], style, labels)
            report[name][style] = result
            rate = f'{result["agreement_pct"]:.1f}' if result['agreement_pct'] is not None else 'n/a'
            print(f'{name:12} {style:8} {len(ids):7} {result["classified"]:11} {result["agreed"]:7} '
                  f'{rate:>8} {result["pending"]:8} {result["missing_records"]:10}')
    output = args.root/'collection-e2v-agreement.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Match% = agreed / classified; pending and missing records are reported separately.')
    print('Saved:', output)


if __name__ == '__main__':
    main()
