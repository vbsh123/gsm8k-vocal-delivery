"""Running aggregates for DIFFA collection; preserve vectors, drift and raw text."""
import csv
import json
from statistics import mean,median
from analyze_paired_drift import vector_metrics
from extract_audio_vectors import GROUPS,save_tensor


def load(plan,folder,config_hash):
    import torch
    path=folder/'collection-stream.pt'
    if not path.exists():return dict(format='diffa-collection-stream-v1',run_config_sha256=config_hash,
        completed_ids=[],rows=[],difference_sum=None,unit_sum=None,valid_count=None)
    state=torch.load(path,map_location='cpu',weights_only=True)
    expected=[p['question_id'] for p in plan['pairs']][:len(state['completed_ids'])]
    if (state['format']!='diffa-collection-stream-v1' or state['run_config_sha256']!=config_hash or
            state['completed_ids']!=expected or [r['question_id'] for r in state['rows']]!=expected):
        raise ValueError('Running collection checkpoint differs from the frozen question prefix.')
    return state


def append(state,saved):
    import torch
    neutral,emotion,difference=saved['neutral'],saved['emotion'],saved['difference']
    if (neutral.shape!=emotion.shape or neutral.shape!=difference.shape or neutral.shape[:2]!=(32,len(GROUPS)) or
            not all(torch.isfinite(x).all() for x in (neutral,emotion,difference))):raise ValueError('Invalid collection means.')
    _,unit,valid,cosine,relative,_=vector_metrics(neutral,emotion,torch)
    delta=difference.double()
    state['difference_sum']=delta.clone() if state['difference_sum'] is None else state['difference_sum']+delta
    state['unit_sum']=unit.clone() if state['unit_sum'] is None else state['unit_sum']+unit
    state['valid_count']=valid.long() if state['valid_count'] is None else state['valid_count']+valid.long()
    row={k:saved[k] for k in ('question_id','neutral_response','emotion_response','neutral_forwards','emotion_forwards',
        'matched_forwards','neutral_encoding','emotion_encoding')}
    row.update(cosine=cosine,relative_change=relative)
    state['rows'].append(row);state['completed_ids'].append(row['question_id'])


def finalize(plan,folder,state):
    import torch
    count=len(plan['pairs'])
    if state['completed_ids']!=[p['question_id'] for p in plan['pairs']]:raise ValueError('Collection incomplete.')
    vector=dict(protocol=plan['protocol'],pair_mean_sums={plan['emotion']:state['difference_sum'].float()},
        pair_counts={plan['emotion']:torch.full((32,),count,dtype=torch.long)},groups=GROUPS,styles=[plan['emotion']],
        prompt=plan['collection_prompt'],max_new_tokens=plan['max_new_tokens'],seed=42,steps=128,mode='fixed',method='mean',
        extraction_question_ids=plan['collection_ids'],sign=f'{plan["emotion"]} minus neutral',averaging=plan['averaging'])
    path=folder/'style-vectors.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if (old['groups']!=GROUPS or old['extraction_question_ids']!=vector['extraction_question_ids'] or old['prompt']!=vector['prompt'] or
                old['max_new_tokens']!=vector['max_new_tokens'] or
                not torch.equal(old['pair_mean_sums'][plan['emotion']],vector['pair_mean_sums'][plan['emotion']]) or
                not torch.equal(old['pair_counts'][plan['emotion']],vector['pair_counts'][plan['emotion']])):
            raise ValueError('Running aggregate differs from existing final vector.')
    else:save_tensor(torch,vector,path)
    rows=[]
    for layer in range(32):
        for g,group in enumerate(GROUPS):
            values=[float(r['relative_change'][layer,g]) for r in state['rows']]
            rows.append(dict(layer=layer,group=group,pairs=count,mean_change_pct=mean(values),median_change_pct=median(values),
                mean_cosine=mean(float(r['cosine'][layer,g]) for r in state['rows']),
                shift_consistency=float(state['unit_sum'][layer,g].norm()/max(1,int(state['valid_count'][layer,g])))))
    with (folder/'layer-drift.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/'collection-results.jsonl').open('w') as stream:
        for pair,saved in zip(plan['pairs'],state['rows']):
            for style in ('neutral','emotion'):
                stream.write(json.dumps(dict(pair[style],response=saved[style+'_response'],prompt=plan['collection_prompt'],
                    split='collection',seed=42,max_new_tokens=plan['max_new_tokens'],forwards=saved[style+'_forwards']))+'\n')
    (folder/'collection-storage.json').write_text(json.dumps(dict(mode='running-aggregates',completed_pairs=count,
        note='Float64 sums of original float32 pair differences; raw responses and per-pair drift scalars retained. '
             'Full per-pair activation tensors are not written for new pairs. Existing pair caches are left untouched.'),indent=2)+'\n')
