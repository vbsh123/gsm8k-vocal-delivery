"""Read DIFFA collection responses for descriptive scoring, without new inference."""
from pathlib import Path

from steer import read_rows


def collection_rows(plan,folder):
    import torch
    from diffa_stream_collection import load
    from winning_diffa_sweeps import digest
    saved=[]
    if (folder/'collection-stream.pt').exists():
        saved=load(plan,folder,digest(folder/'run_config.json'))['rows']
    else:
        for i,pair in enumerate(plan['pairs']):
            path=folder/'pairs'/f'{i:05d}.pt'
            if not path.exists():continue
            row=torch.load(path,map_location='cpu',weights_only=True)
            if row['question_id']!=pair['question_id'] or row['run_config_sha256']!=digest(folder/'run_config.json'):
                raise ValueError('Collection answer cache differs from frozen configuration.')
            saved.append(row)
    pairs={p['question_id']:p for p in plan['pairs']};rows=[]
    for item in saved:
        pair=pairs[item['question_id']]
        for key,label in (('neutral','neutral'),('emotion',plan['emotion'])):
            record=pair[key]
            rows.append(dict(record,condition=label,response=item[key+'_response'],
                reference_answer=record.get('reference_answer',record.get('answer','')),prompt=plan['collection_prompt'],
                seed=42,max_new_tokens=plan['max_new_tokens'],steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                strength=0.,interventions=[],split='collection',forwards=item[key+'_forwards'],audio_encoding=item[key+'_encoding']))
    if not rows:
        for row in read_rows(folder/'collection-results.jsonl'):
            if row['question_id'] not in pairs or row['prompt']!=plan['collection_prompt'] or row['max_new_tokens']!=plan['max_new_tokens']:
                raise ValueError('Exported collection response differs.')
            rows.append(dict(row,condition=row.get('condition',row.get('style')),reference_answer=row.get('reference_answer',row.get('answer','')),strength=0.))
    return rows


def collection_bank(plan):
    from diffa_happy_nightly import normal_answer
    task=plan['name'];bank=dict(plan['bank']);ids=set(plan['collection_ids'])
    if task not in ('bbh3','bbh5','ifeval'):return bank
    required={'original_input','original_target'} if task.startswith('bbh') else {'key','prompt','instruction_id_list','kwargs'}
    if all(required<=bank[q].keys() for q in ids):return bank
    if task=='ifeval':
        import run_spoken_ifeval as official
        source=official.setup_official()/'data/input_data.jsonl'
        examples={f'ifeval_{q["key"]}':q for q in read_rows(source)}
        for q in ids:
            example=examples[q]
            if example['prompt']!=bank[q]['question']:raise ValueError('Official IFEval collection prompt differs.')
            bank[q]=dict(bank[q],**example)
        return bank
    parents={Path(p[s]['audio_file']).parent for p in plan['pairs'] for s in ('neutral','emotion')}
    paths=[p/'questions.jsonl' for p in sorted(parents)]
    if plan['target'].get('bank_file'):paths.insert(0,Path(plan['target']['bank_file']))
    def add(questions):
        for q in questions:
            qid=q.get('question_id')
            if qid not in ids or not required<=q.keys():continue
            if (q.get('question',bank[qid]['question'])!=bank[qid]['question'] or
                    normal_answer(q['original_target']).strip('()')!=normal_answer(bank[qid]['answer']).strip('()')):
                raise ValueError('Original BBH collection source differs from frozen audio/reference.')
            bank[qid]=dict(bank[qid],**q)
    for path in paths:add(read_rows(path))
    if not all(required<=bank[q].keys() for q in ids):
        filename='logical_deduction_'+('three' if task=='bbh3' else 'five')+'_objects-source.json'
        for parent in sorted(parents):
            path=parent/filename
            if not path.exists():continue
            if task=='bbh3':
                from run_bbh_logical200 import select_questions,SOURCE_SHA256
                from winning_diffa_sweeps import digest
                if digest(path)!=SOURCE_SHA256:raise ValueError('Pinned BBH source hash differs.')
                add(select_questions(path.read_bytes(),250))
            else:
                from run_reasoning_transfer import bbh_questions
                from run_kimi_bbh5 import SOURCE_SHA256
                from winning_diffa_sweeps import digest
                if digest(path)!=SOURCE_SHA256:raise ValueError('Pinned BBH source hash differs.')
                add(bbh_questions(path.read_bytes()))
    missing=[q for q in ids if not required<=bank[q].keys()]
    if missing:raise FileNotFoundError(f'Restore original BBH question metadata for collection scoring: {missing[:3]}')
    return bank
