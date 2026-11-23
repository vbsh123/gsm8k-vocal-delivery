"""Report actual saved neutral/emotion decoder forward counts. CPU only, read only."""
import argparse
import json
from pathlib import Path
from statistics import mean,median

DEFAULTS=(
    'outputs/diffa-gsm-vanilla/angry/agreed/collection-stream.pt',
    'outputs/diffa-gsm-vanilla/angry/random/collection-stream.pt',
    'outputs/diffa-active-pooling-fit200-eval200/collection-checkpoint.pt',
    'outputs/diffa-full-buffer-fit200-eval200-all-forwards/collection-checkpoint.pt',
)


def statistics(rows):
    if not rows:
        return None
    neutral=[r['neutral_forwards'] for r in rows]
    emotion=[r['emotion_forwards'] for r in rows]
    if any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in neutral+emotion):
        raise ValueError('Forward counts must be positive integers.')
    hypothetical=[100*min(n,e)/e for n,e in zip(neutral,emotion)]
    saved=[r.get('matched_forwards') for r in rows]
    if all(v is None for v in saved):
        mode='all forwards (no prefix truncation)'
        actual=100.
    elif all(isinstance(v,int) and v==min(n,e) for v,n,e in zip(saved,neutral,emotion)):
        mode='matched chronological prefixes'
        actual=mean(hypothetical)
    else:
        raise ValueError('Saved time-averaging metadata is inconsistent.')
    describe=lambda values:dict(mean=mean(values),median=median(values),minimum=min(values),maximum=max(values))
    return dict(pairs=len(rows),neutral=describe(neutral),emotion=describe(emotion),mode=mode,
        angry_longer=sum(e>n for n,e in zip(neutral,emotion)),
        hypothetical_matched_angry_retained_mean_pct=mean(hypothetical),
        hypothetical_matched_angry_retained_median_pct=median(hypothetical),
        actual_angry_retained_mean_pct=actual)


def original_rows(config_path):
    import torch
    config=json.loads(config_path.read_text())
    vector=Path(config['vector'])
    state=torch.load(vector,map_location='cpu',weights_only=True)
    pairs=state.get('pairs')
    ids=[p['question_id'] for p in pairs] if pairs else state.get('extraction_question_ids',[])
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('Original vector lacks unique contributing question IDs.')
    candidates=[folder for folder in (vector.parent/'pairs',vector.parent.parent/'pairs') if folder.is_dir()]
    if len(candidates)!=1:
        raise ValueError(f'Need the original per-pair forward-count caches beside {vector}; found {candidates}. No inference performed.')
    print(f'Original vector: {vector}\nCollection prompt: {state.get("prompt")}')
    print(f'Reading saved pair caches from {candidates[0]}; CPU only.',flush=True)
    expected=set(ids)
    selected={}
    for path in sorted(candidates[0].glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        q=saved.get('question_id') or saved.get('pair',{}).get('question_id')
        if q not in expected:
            continue
        if q in selected:
            raise ValueError(f'Duplicate original fitting pair: {q}')
        selected[q]=dict(question_id=q,neutral_forwards=saved['neutral_forwards'],
            emotion_forwards=saved['emotion_forwards'] if 'emotion_forwards' in saved else saved['angry_forwards'],
            matched_forwards=saved['matched_forwards'])
    if set(selected)!=expected:
        raise ValueError(f'Missing original forward-count caches: {len(expected-set(selected))}/{len(ids)} pairs.')
    return [selected[q] for q in ids]


def report(path,original=False):
    import torch
    if original:
        rows=original_rows(path)
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
        # Both pooling scopes came from the same generation; count each pair once.
        saved=state['scopes'].get('full-buffer',next(iter(state['scopes'].values()))) if 'scopes' in state else state
        rows=saved['rows']
    if len({r['question_id'] for r in rows})!=len(rows):
        raise ValueError(f'Duplicate fitting questions: {path}')
    stats=statistics(rows)
    print(f'\n{path}')
    if stats is None:
        print('No completed pairs saved yet.')
        return
    print(f'{stats["pairs"]} completed pairs; {stats["mode"]}.')
    print('Condition       Mean forwards  Median  Min  Max')
    for label,key in [('Neutral','neutral'),('Angry','emotion')]:
        value=stats[key]
        print(f'{label:12} {value["mean"]:16.2f} {value["median"]:7g} {value["minimum"]:4} {value["maximum"]:4}')
    print(f'Angry used more forwards: {stats["angry_longer"]}/{stats["pairs"]} pairs.')
    print(f'Matched-prefix averaging would retain angry: mean={stats["hypothetical_matched_angry_retained_mean_pct"]:.1f}%; '
          f'median={stats["hypothetical_matched_angry_retained_median_pct"]:.1f}% (computed per pair).')
    print(f'This checkpoint actually retains angry: mean={stats["actual_angry_retained_mean_pct"]:.1f}%.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints',nargs='*',type=Path)
    parser.add_argument('--original-gsm',action='store_true',help='Read the original vector and its per-pair caches via the historical run config.')
    parser.add_argument('--original-config',type=Path,default=Path('outputs/reasoning-transfer/gsm8k/run_config.json'))
    args=parser.parse_args()
    if args.original_gsm:
        report(args.original_config,original=True)
        return
    paths=args.checkpoints or [Path(p) for p in DEFAULTS if Path(p).is_file()]
    print('Saved model forward counts, NOT output token counts. No inference or file changes.')
    if not paths:
        print('No known collection checkpoints found on this machine yet.')
    for path in paths:
        report(path)


if __name__=='__main__':
    main()
