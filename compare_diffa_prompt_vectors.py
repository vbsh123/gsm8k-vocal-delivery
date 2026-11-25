"""Compare the saved answer-position steering directions under two collection prompts on CPU."""
import argparse
import csv
import json
from pathlib import Path

from compare_diffa_emotion_vectors import TASKS, direction, geometry, load_vector


def original_path(task, emotion):
    if task in ('bbh3', 'bbh5'):
        return Path('outputs/emotion-bbh-original-prompt/diffa')/task/emotion/'style-vectors.pt'
    if emotion != 'angry':
        return Path(f'outputs/diffa-{emotion}-winners')/task/'style-vectors.pt'
    if task == 'gsm8k':
        return Path('outputs/jenny-angry-vectors/style-vectors.pt')
    if task == 'multiarith':
        return Path('outputs/multiarith580-angry-mean/style-vectors.pt')
    catalog = json.loads(Path(__file__).with_name('winning_diffa_catalog.json').read_text())
    folder = Path(catalog[task]['run'])
    from winning_diffa_sweeps import config_file, vector_path
    return vector_path(folder, json.loads(config_file(folder).read_text()))


def report(task, emotion, original, bare, output):
    data = [load_vector(path, emotion, mmap=True) for path in (original, bare)]
    for label, item in zip(('original', 'bare'), data):
        state = item['state']
        print(f'{label}: {item["path"]}\n  collection prompt: {state.get("prompt")!r}\n'
              f'  pooling: {state.get("averaging", "unspecified in checkpoint")}')
    a, b = data
    if a['sums'].shape[0] != b['sums'].shape[0] or a['sums'].shape[-1] != b['sums'].shape[-1]:
        raise ValueError('Layer counts or hidden widths differ.')
    overlap = None
    if a['ids'] is not None and b['ids'] is not None:
        overlap = len(set(a['ids']) & set(b['ids']))
    rows = []
    for layer in range(a['sums'].shape[0]):
        x, y = (direction(d, layer, 'answer') for d in data)
        if x is None or y is None:
            continue
        g = geometry(x, y)
        row = dict(dataset=task, emotion=emotion, layer=layer, group='answer',
                   original_pairs=int(a['counts'][layer]), bare_pairs=int(b['counts'][layer]),
                   overlapping_question_ids=overlap, cosine=g['cosine'], angle_degrees=g['angle_degrees'],
                   original_norm=g['angry_norm'], bare_norm=g['happy_norm'],
                   bare_over_original_norm=g['happy_over_angry_norm'])
        rows.append(row)
    if not rows:
        raise ValueError('No common fitted answer layers.')
    dest = output/f'{task}-{emotion}.csv'
    output.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{task} / {emotion}: {overlap} shared fitting IDs. Full layer table: {dest}')
    for r in rows:
        if r['layer'] in (15, 25, 28, 30, 31):
            print('L{layer}: cosine={cosine}, bare/original norm={bare_over_original_norm} '
                  '(pairs {original_pairs}/{bare_pairs})'.format(**r))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=TASKS)
    p.add_argument('--emotion', choices=('angry', 'happy', 'sad'), default='angry')
    p.add_argument('--original', type=Path)
    p.add_argument('--bare', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/prompt-vector-comparison'))
    args = p.parse_args()
    bare = args.bare or Path('outputs/emotion-block-night/diffa')/args.dataset/args.emotion/'style-vectors.pt'
    original = args.original or original_path(args.dataset, args.emotion)
    for path in (original, bare):
        if not path.is_file():
            p.exit(1, f'Missing saved vector: {path}\n')
    report(args.dataset, args.emotion, original, bare, args.output)


if __name__ == '__main__':
    main()
