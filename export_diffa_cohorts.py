"""Export exact original DIFFA collection/evaluation cohorts for the fresh Phi run; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = 'original-diffa-cross-model-cohorts-v1'
DATASETS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')


def make_cohort(plan, bank):
    train, test = plan['collection_ids'], plan['test_ids']
    if (not train or not test or len(set(train+test)) != len(train+test)
            or [p['question_id'] for p in plan['pairs']] != train
            or [r['question_id'] for r in plan['target']['records']] != test):
        raise ValueError('DIFFA collection/target IDs differ from the frozen plan or overlap.')
    neutral = {p['question_id']: p['neutral'] for p in plan['pairs']}
    neutral.update({r['question_id']: r for r in plan['target']['records']})
    if not set(train+test) <= bank.keys() or not all(r.get('audio_sha256') for r in neutral.values()):
        raise ValueError('Missing question metadata or exact neutral audio hashes.')
    return dict(task=plan['name'], seed=plan['seed'], collection_ids=train, test_ids=test,
                prompt=plan['target']['prompt'], collection_prompt=plan['collection_prompt'],
                max_new_tokens=plan['target']['max_new_tokens'],
                collection_max_new_tokens=plan['max_new_tokens'],
                bank={q:bank[q] for q in train+test}, neutral=neutral,
                selection='Exact original DIFFA contributing collection and held-out evaluation IDs; no new selection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--output', type=Path, default=Path('/workspace/diffa-original-cohorts.json'))
    args = parser.parse_args()
    from diffa_collection_report import collection_bank
    result = dict(protocol=PROTOCOL, datasets={})
    for name in DATASETS:
        path = args.root/name/'run_config.json'
        plan = json.loads(path.read_text())
        if plan['name'] != name or plan['seed'] != 42:
            raise ValueError('Unexpected original DIFFA task or seed.')
        bank = collection_bank(plan)
        # Held-out bank retains the official scorer's original reference metadata.
        bank.update(plan['target']['bank'])
        cohort = make_cohort(plan, bank)
        cohort['source_plan_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['datasets'][name] = cohort
        print(f'{name}: EXACT original DIFFA {len(cohort["collection_ids"])}/{len(cohort["test_ids"])} collection/evaluation IDs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('COPY TO PHI:', args.output)
    print(f'Size: {args.output.stat().st_size/2**20:.2f} MiB; no audio, activations or model weights.')


if __name__ == '__main__':
    main()
