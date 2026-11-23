"""Read-only angry/happy DIFFA vector geometry and saved direction consistency; CPU only."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch


TASKS=('gsm8k','svamp','multiarith','bbh3','bbh5','ifeval')


def geometry(angry,happy):
    a,h=angry.double().reshape(-1),happy.double().reshape(-1)
    if a.shape!=h.shape or not torch.isfinite(a).all() or not torch.isfinite(h).all():
        raise ValueError('Vector widths differ or contain nonfinite values.')
    an,hn=float(a.norm()),float(h.norm())
    result=dict(angry_norm=an,happy_norm=hn,happy_over_angry_norm=hn/an if an else None,
                relative_distance_to_angry=float((h-a).norm())/an if an else None,
                cosine=None,angle_degrees=None,happy_parallel_coefficient=None,happy_orthogonal_squared_fraction=None)
    if not an or not hn:return result
    cosine=max(-1.,min(1.,float(torch.dot(a,h))/(an*hn)))
    result.update(cosine=cosine,angle_degrees=math.degrees(math.acos(cosine)),
                  happy_parallel_coefficient=float(torch.dot(a,h))/(an*an),
                  happy_orthogonal_squared_fraction=max(0.,1.-cosine*cosine))
    return result


def load_vector(path,emotion):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('method','mean')!='mean' or state.get('mode','fixed')!='fixed':
        raise ValueError('Compare the fixed raw mean vectors, not normalized/probe/SVD alternatives.')
    sums=state['pair_mean_sums'][emotion].float()
    counts=state['pair_counts'][emotion]
    if sums.ndim!=3 or len(counts)!=sums.shape[0] or len(state['groups'])!=sums.shape[1]:
        raise ValueError('Invalid stored vector axes/counts.')
    count=int(counts[31])
    ids=list(state.get('extraction_question_ids',[]))
    if len(ids)!=count:
        actual=[p['question_id'] for p in state.get('pairs',[])]
        ids=actual if len(actual)==count else None
    if ids is not None and len(set(ids))!=len(ids):raise ValueError('Duplicate fitting IDs.')
    return dict(state=state,sums=sums,counts=counts,ids=ids,path=path,emotion=emotion,
                targets={tuple(x) for x in state.get('collected_targets',[])})


def direction(data,layer,group):
    count=int(data['counts'][layer])
    if not count or data['targets'] and (layer,group) not in data['targets']:return None
    # Match the actual steering reader: float32 saved sum / count, then double for geometry.
    return data['sums'][layer,data['state']['groups'].index(group)]/count


def consistency(data,folder):
    path=folder/'collection-stream.pt'
    if not path.exists():return None
    state=torch.load(path,map_location='cpu',weights_only=True)
    ids=data['ids']
    if ids is None or set(state.get('completed_ids',[]))!=set(ids) or len(state['completed_ids'])!=len(ids):return None
    if state.get('unit_sum') is None or state.get('valid_count') is None:return None
    return state


def compare(angry,happy,angry_folder,happy_folder):
    if angry['sums'].shape[0]!=happy['sums'].shape[0] or angry['sums'].shape[-1]!=happy['sums'].shape[-1]:
        raise ValueError('Layer count or hidden width differs.')
    same_ids=angry['ids'] is not None and happy['ids'] is not None and set(angry['ids'])==set(happy['ids'])
    fields=('prompt','max_new_tokens','seed','steps','averaging')
    metadata={e:{k:d['state'].get(k) for k in fields} for e,d in [('angry',angry),('happy',happy)]}
    same_prompt=metadata['angry']['prompt']==metadata['happy']['prompt']
    same_budget=metadata['angry']['max_new_tokens']==metadata['happy']['max_new_tokens']
    recipe_mismatches=[k for k in ('seed','steps') if metadata['angry'][k] is not None and metadata['happy'][k] is not None
                       and metadata['angry'][k]!=metadata['happy'][k]]
    streams={e:consistency(d,folder) for e,d,folder in [('angry',angry,angry_folder),('happy',happy,happy_folder)]}
    rows=[]
    groups=[g for g in angry['state']['groups'] if g in happy['state']['groups']]
    for layer in range(angry['sums'].shape[0]):
        for group in groups:
            a,h=direction(angry,layer,group),direction(happy,layer,group)
            if a is None or h is None:continue
            row=dict(layer=layer,group=group,angry_pairs=int(angry['counts'][layer]),happy_pairs=int(happy['counts'][layer]),
                     same_fitting_questions=same_ids,same_collection_prompt=same_prompt,same_budget=same_budget,**geometry(a,h))
            for e,d in [('angry',angry),('happy',happy)]:
                state=streams[e]
                value=None
                if state is not None:
                    g=d['state']['groups'].index(group)
                    n=int(state['valid_count'][layer,g])
                    if n:value=float(state['unit_sum'][layer,g].norm())/n
                row[e+'_direction_consistency']=value
            rows.append(row)
    return dict(metadata=metadata,same_fitting_questions=same_ids,same_collection_prompt=same_prompt,same_budget=same_budget,
                known_generation_recipe_mismatches=recipe_mismatches,
                angry_fitting_count=len(angry['ids']) if angry['ids'] is not None else None,
                happy_fitting_count=len(happy['ids']) if happy['ids'] is not None else None,rows=rows,
                per_question_cross_emotion_cosine='Unavailable from aggregate-only checkpoints; no per-question vectors reconstructed.',
                note='Geometry of actual stored mean steering vectors. Different geometry does not establish different behavioral effects. '
                     'Direction consistency is length of mean unit per-question shift (0 dispersed/cancelling,1 aligned), not uncertainty of the final vector.')


def paths(args,name):
    if args.cohort=='original':
        folder=args.original_root/name
        plan=json.loads((folder/'run_config.json').read_text())
        return Path(plan['target']['vector']),folder/'style-vectors.pt',Path(plan['target']['vector']).parent,folder
    if args.cohort=='shared3':
        folder=args.shared_gsm if name=='gsm8k' else args.shared_root/name
    else:
        folder=args.refit_root/args.cohort.removeprefix('step-')/name
    return folder/'angry/style-vectors.pt',folder/'happy/style-vectors.pt',folder/'angry',folder/'happy'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=('original','shared3','step-original','step-shared3'),default='original')
    parser.add_argument('--datasets',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--layers',type=int,nargs='+',default=[31])
    parser.add_argument('--groups',nargs='+',default=['answer'])
    parser.add_argument('--original-root',type=Path,default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--shared-gsm',type=Path,default=Path('outputs/diffa-gsm-shared-three-zeroshot-hashes'))
    parser.add_argument('--shared-root',type=Path,default=Path('outputs/diffa-shared-three-tasks'))
    parser.add_argument('--refit-root',type=Path,default=Path('outputs/diffa-step-by-step-refit'))
    parser.add_argument('--output',type=Path,default=Path('outputs/emotion-vector-comparison'))
    args=parser.parse_args()
    results={};table=[]
    print('CPU only. Actual angry-minus-neutral versus happy-minus-neutral means; no inference or changes to experiments.')
    print(f'{"Dataset":12} {"Layer/group":15} {"Cosine":>9} {"Angle°":>9} {"||H||/||A||":>13} {"||H-A||/||A||":>16}')
    for name in args.datasets:
        try:
            a,h,af,hf=paths(args,name)
            if not a.exists() or not h.exists():
                print(f'{name}: pending/missing completed vector: {a if not a.exists() else h}');continue
            result=compare(load_vector(a,'angry'),load_vector(h,'happy'),af,hf)
        except FileNotFoundError:
            print(name,'cohort not prepared on this machine.');continue
        results[name]=result
        for row in result['rows']:
            table.append(dict(dataset=name,**row))
            if row['layer'] not in args.layers or row['group'] not in args.groups:continue
            def value(k):return f'{row[k]:.4f}' if row[k] is not None else 'n/a'
            print(f'{name:12} {str(row["layer"])+"/"+row["group"]:15} {value("cosine"):>9} {value("angle_degrees"):>9} '
                  f'{value("happy_over_angry_norm"):>13} {value("relative_distance_to_angry"):>16}')
        if result['known_generation_recipe_mismatches'] or not all(result[k] for k in ('same_fitting_questions','same_collection_prompt','same_budget')):
            print('  CAUTION: fitting IDs, prompt or budget differ; see JSON provenance before interpreting emotion specificity.')
    output=args.output/args.cohort;output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if table:
        with (output/'all-layers.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('Cosine: 1 same direction,0 perpendicular,-1 opposite. Norm ratio measures magnitude; distance includes both angle and magnitude.')
    print('All available layer/group geometry and saved direction-consistency statistics:',output)


if __name__=='__main__':main()
