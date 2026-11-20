"""DIFFA GSM8K emotion-minus-neutral collection, drift and held-out steering.

Reuse the original 500 excluded collection IDs and 718 held-out questions.
The aligned #### prompt is used for BOTH collection and evaluation; no label filtering.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import sys
import time

from extract_audio_vectors import GROUPS, average_pair, save_tensor
from run_reasoning_transfer import HASH_PROMPT, digest, freeze, generate_audio, recording, NumericScorer
from steer import install_interventions, read_rows

PROTOCOL = 'diffa-gsm-emotion-matched-forward-v1'
VOICE = 'en-US-JennyNeural'
AUDIO_SOURCES = [Path('outputs')/name for name in ('azure-jenny500-neutral', 'azure-jenny500-extra',
    'azure-jenny-next500', 'azure-jenny-remaining-test')]


def select_ids(previous, vector):
    train = sorted(set(vector['extraction_question_ids']))
    test = [r for r in previous['recordings'] if r['condition'] == 'neutral']
    ids = [r['question_id'] for r in test]
    if len(train) != 500 or len(ids) != 718 or len(set(ids)) != 718 or set(train) & set(ids):
        raise ValueError('Expected the original 500 excluded collection IDs / 718 disjoint test IDs.')
    if previous['prompt'] != HASH_PROMPT or any(r['voice'] != VOICE for r in test):
        raise ValueError('Expected the aligned #### Jenny GSM8K baseline.')
    return train, test


def scan_audio(sources, bank):
    found = {}
    for source in sources:
        for row in read_rows(source/'plan.jsonl'):
            qid = row['question_id']; style = row['style']
            if row.get('provider') != 'azure' or row['voice'] != VOICE or qid not in bank or style not in ('neutral', 'happy', 'sad'):
                continue
            gold = str(row['answer']).rsplit('####', 1)[-1].strip()
            if row['question'] != bank[qid]['question'] or gold != bank[qid]['answer']:
                raise ValueError(f'Audio text/reference mismatch: {source}, {qid}')
            if not (source/row['audio_path']).is_file():
                continue
            found.setdefault((qid, style), (row, source))
    return found


def prepare(args):
    import torch
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    path = root/'run_config.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['protocol'] != PROTOCOL or plan['emotion'] != args.emotion or plan['source_run'] != str(args.benchmark.absolute()):
            raise ValueError('Output belongs to another frozen experiment.')
        return plan
    previous = json.loads((args.benchmark/'run_config.json').read_text())
    vector_path = Path(previous['vector'])
    if digest(vector_path) != previous['vector_sha256']:
        raise ValueError('Original vector changed; cannot establish the original collection exclusions.')
    old = torch.load(vector_path, map_location='cpu', weights_only=True)
    train, test = select_ids(previous, old)
    official = read_rows(args.benchmark/'gsm8k-test.jsonl')
    if not official:
        raise FileNotFoundError(f'Missing official question source: {args.benchmark}/gsm8k-test.jsonl')
    bank = {}
    for qid in train+[r['question_id'] for r in test]:
        q = official[int(qid.rsplit('_', 1)[1])]
        bank[qid] = dict(question_id=qid, question=q['question'], answer=q['answer'].rsplit('####', 1)[-1].strip())
    if any(previous['bank'][r['question_id']]['question'] != bank[r['question_id']]['question'] or
           previous['bank'][r['question_id']]['answer'] != bank[r['question_id']]['answer'] for r in test):
        raise ValueError('Official test source differs from the frozen evaluation bank.')
    generated = [root/'audio'/s for s in ('neutral', args.emotion)]
    sources = args.audio_sources+generated
    found = scan_audio(sources, bank)
    # A new neutral recording cannot replace the frozen held-out baseline.
    for row in test:
        key=row['question_id'],'neutral'
        if key not in found:
            original=Path(row['audio_file'])
            if not original.is_file():raise FileNotFoundError(f'Restore original held-out neutral audio: {original}')
            q=bank[row['question_id']]
            found[key]=(dict(q,provider='azure',voice=VOICE,style='neutral',audio_path=original.name),original.parent)
        if digest(found[key][1]/found[key][0]['audio_path'])!=row['audio_sha256']:
            raise ValueError('Held-out neutral audio changed; no TTS requested.')
    original_test_neutrals={key:value for key,value in found.items() if key[1]=='neutral' and key[0] in {r['question_id'] for r in test}}
    for style, destination in zip(('neutral', args.emotion), generated):
        missing = [q for q in bank if (q, style) not in found]
        if missing:
            # Keep the first request bank unchanged on resume, even after some clips exist.
            existing = read_rows(destination/'questions.jsonl')
            questions = existing or [bank[q] for q in missing]
            print(f'Audio: {len(missing)} missing {style} clips; resume {destination}', flush=True)
            generate_audio(destination, questions, [style])
    found = scan_audio(sources, bank)
    found.update(original_test_neutrals)
    missing = [(q, s) for q in bank for s in ('neutral', args.emotion) if (q, s) not in found]
    if missing:
        raise ValueError(f'Missing audio after TTS: {missing[:5]}')
    records = {key: recording(*value) for key, value in found.items() if key[1] in ('neutral', args.emotion)}
    for r in test:
        if records[r['question_id'], 'neutral']['audio_sha256'] != r['audio_sha256']:
            raise ValueError('Held-out neutral audio differs from the original baseline.')
    plan = dict(protocol=PROTOCOL, model='zhoujiaming777/DIFFA-2', emotion=args.emotion,
        source_run=str(args.benchmark.absolute()), source_config_sha256=digest(args.benchmark/'run_config.json'),
        original_vector_sha256=digest(vector_path), collection_ids=train,
        test_ids=[r['question_id'] for r in test], bank=bank,
        pairs=[dict(question_id=q, neutral=records[q, 'neutral'], emotion=records[q, args.emotion]) for q in train],
        recordings=[dict(records[q, s], condition=s) for q in [r['question_id'] for r in test] for s in ('neutral', args.emotion)],
        prompt=HASH_PROMPT, collection_prompt=HASH_PROMPT, seed=42, max_new_tokens=512, steps=128,
        block_length=32, temperature=0., decoder='fast_dllm', groups=GROUPS, layers=list(range(32)),
        averaging='Mean over token positions and matched decoding forwards per pair; then equal question weight.',
        selection='Original500 excluded IDs, no outcome/classifier filtering; original718 held-out IDs.',
        note='New emotion vector uses all500 pairs and aligned collection prompt. Historical angry vector used425 classifier-selected pairs and Final answer collection prompt; not an isolated emotion-only comparison.',
        audio_encoding='Native <=30s; longer full audio through chunked Whisper and global adapters; no cropping.')
    freeze(path, plan)
    print(f'{len(train)} collection pairs; {len(test)} held-out neutral/{args.emotion} pairs. No classifier filtering.', flush=True)
    return plan


def capture(model, generate, decoder, max_new_tokens=512):
    """Observe all block/group means during the unchanged full native decoding run."""
    import torch
    layers = model.llm_model.model.transformer.blocks
    traces = [[] for _ in layers]; bounds = {}; handles = []
    original = decoder.insert_audio_embedding
    def insert(text, speech, lengths):
        start = int(lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size, end=start+speech.shape[1])
        return original(text, speech, lengths)
    def hook(index):
        def observe(module, inputs, result):
            h = result[0]
            spans = [slice(bounds['start'], bounds['acoustic_end']), slice(bounds['acoustic_end'], bounds['end']),
                     slice(bounds['start'], bounds['end']), slice(-max_new_tokens, None)]
            traces[index].append(torch.stack([h[0, span].detach().float().mean(0).cpu() for span in spans]))
        return observe
    decoder.insert_audio_embedding = insert
    try:
        handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(layers)]
        response, encoding = generate()
    finally:
        decoder.insert_audio_embedding = original
        for handle in handles:handle.remove()
    return response, encoding, torch.stack([torch.stack(t) for t in traces])


def seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def generate(record, plan, model, tokenizer, features):
    import soundfile as sf
    from extract_vectors import gen_chunked
    if digest(record['audio_file']) != record['audio_sha256']:
        raise ValueError('Frozen audio changed.')
    audio, rate = sf.read(record['audio_file'], dtype='float32')
    seed()
    return gen_chunked(audio, rate, model, tokenizer, features, plan['prompt'], 512, 30)


def collect(plan, root, model, tokenizer, features):
    import torch
    import src.fast_dllm.diffa_generate as decoder
    pairs = root/'pairs'; pairs.mkdir(exist_ok=True)
    for i, pair in enumerate(plan['pairs']):
        path = pairs/f'{i:05d}.pt'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['question_id'] != pair['question_id'] or saved['protocol'] != PROTOCOL:
                raise ValueError('Pair cache differs from frozen collection.')
            continue
        print(f'Collect [{i+1}/{len(plan["pairs"])}] {pair["question_id"]}: neutral + {plan["emotion"]}', flush=True)
        outputs = []
        for style in ('neutral', 'emotion'):
            record = pair[style]
            outputs.append(capture(model, lambda: generate(record, plan, model, tokenizer, features), decoder))
        nr, ne, n = outputs[0]; er, ee, h = outputs[1]
        nm, hm, diff, matched = average_pair(n, h, 'matched')
        save_tensor(torch, dict(protocol=PROTOCOL, question_id=pair['question_id'], neutral=nm,
            emotion=hm, difference=diff, neutral_response=nr, emotion_response=er,
            neutral_forwards=n.shape[1], emotion_forwards=h.shape[1], matched_forwards=matched,
            neutral_encoding=ne, emotion_encoding=ee), path)
        del outputs, n, h, nm, hm, diff
    fit(plan, root)
    export_collection_answers(plan,root)


def export_collection_answers(plan,root):
    import torch
    path=root/'collection-results.jsonl'
    with path.open('w') as stream:
        for i,pair in enumerate(plan['pairs']):
            saved=torch.load(root/'pairs'/f'{i:05d}.pt',map_location='cpu',weights_only=True)
            for key in ('neutral','emotion'):
                record=pair[key]
                row=dict(record,condition=record['style'],response=saved[key+'_response'],
                    reference_answer=record['answer'],prompt=plan['prompt'],seed=42,max_new_tokens=512,
                    steps=128,block_length=32,temperature=0.,decoder='fast_dllm',interventions=[],
                    split='collection',forwards=saved[key+'_forwards'])
                stream.write(json.dumps(row)+'\n')


def fit(plan, root):
    import torch
    vector_file = root/'style-vectors.pt'
    if vector_file.exists():return
    sums = None
    for i, pair in enumerate(plan['pairs']):
        saved = torch.load(root/'pairs'/f'{i:05d}.pt', map_location='cpu', weights_only=True)
        if saved['question_id'] != pair['question_id']:raise ValueError('Pair identity mismatch.')
        difference = saved['difference'].double()
        sums = difference.clone() if sums is None else sums+difference
    emotion = plan['emotion']; count = len(plan['pairs'])
    state = dict(protocol=plan.get('protocol',PROTOCOL), pair_mean_sums={emotion: sums.float()},
        pair_counts={emotion: torch.full((32,), count, dtype=torch.long)}, groups=GROUPS, styles=[emotion],
        prompt=plan.get('collection_prompt',plan['prompt']), max_new_tokens=plan.get('max_new_tokens',512), seed=42, steps=128, mode='fixed', method='mean',
        extraction_question_ids=plan['collection_ids'], sign=f'{emotion} minus neutral', averaging=plan['averaging'])
    save_tensor(torch, state, vector_file)
    analysis(plan, root)


def analysis(plan, root):
    import torch
    from analyze_paired_drift import vector_metrics
    values = []; units = None; valid_count = None
    for i in range(len(plan['pairs'])):
        saved = torch.load(root/'pairs'/f'{i:05d}.pt', map_location='cpu', weights_only=True)
        delta, unit, valid, cos, rel, norm = vector_metrics(saved['neutral'], saved['emotion'], torch)
        units = unit.clone() if units is None else units+unit
        valid_count = valid.long() if valid_count is None else valid_count+valid
        values.append((cos, rel))
    rows = []
    for layer in range(32):
        for g, group in enumerate(GROUPS):
            rows.append(dict(layer=layer, group=group, pairs=len(values),
                mean_change_pct=mean(float(v[1][layer,g]) for v in values),
                median_change_pct=median(float(v[1][layer,g]) for v in values),
                mean_cosine=mean(float(v[0][layer,g]) for v in values),
                shift_consistency=float(units[layer,g].norm()/max(1,int(valid_count[layer,g])))))
    with (root/'layer-drift.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved all32 layers ×4 token groups: style-vectors.pt and layer-drift.csv.', flush=True)


def evaluation_plan(plan, strengths, root):
    strengths = list(dict.fromkeys(float(a) for a in strengths if a != 0))
    if not strengths or not all(math.isfinite(a) for a in strengths):raise ValueError('Need finite nonzero steering strengths.')
    tasks = list(plan['recordings'])
    for row in plan['recordings']:
        if row['condition'] == 'neutral':
            tasks.extend(dict(row, condition=f'neutral-{plan["emotion"]}-L31-answer-a{a:g}', strength=a) for a in strengths)
    return dict(plan, recordings=tasks, strengths=strengths, vector=str((root/'style-vectors.pt').absolute()))


def reusable(row, task, plan):
    strength=task.get('strength',0.)
    expected=[dict(layer=31,group='answer',strength=strength)] if strength else []
    if (row.get('interventions') or []) != expected:return False
    if strength and (not Path(plan['vector']).exists() or row.get('vector_sha256')!=digest(plan['vector'])):return False
    return (row.get('prompt') == plan['prompt']
        and row.get('audio_sha256') == task['audio_sha256']
        and row.get('reference_answer') == task['answer']
        and all(row.get(k) == plan[k] for k in ('seed','max_new_tokens','steps','block_length','decoder'))
        and row.get('temperature', 0.) == 0.)


def reuse(plan, folder, sources):
    tasks = {(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']}
    done = {(r['question_id'],r['voice'],r['condition']) for r in read_rows(folder/'results.jsonl',repair=True)}
    for source in sources:
        for row in read_rows(source/'results.jsonl'):
            key = row['question_id'], row.get('voice',row.get('speaker')), row.get('condition',row.get('style'))
            if key in done or key not in tasks:continue
            if not reusable(row,tasks[key],plan):continue
            with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(dict(row,reused_from=str(source)))+'\n')
            done.add(key)
    print(f'{len(done)}/{len(tasks)} answers available after exact-recipe reuse.',flush=True)


def evaluate(plan, folder, root, model, tokenizer, features):
    import torch
    state = torch.load(root/'style-vectors.pt', map_location='cpu', weights_only=True)
    emotion = plan['emotion']; n = int(state['pair_counts'][emotion][31])
    vector = state['pair_mean_sums'][emotion][31, GROUPS.index('answer')]/n
    vector = vector.to(device='cuda',dtype=torch.bfloat16)
    vector_hash = digest(root/'style-vectors.pt')
    layers = model.llm_model.model.transformer.blocks
    done = {(r['question_id'],r['voice'],r['condition']) for r in read_rows(folder/'results.jsonl',repair=True)}
    for i, task in enumerate(plan['recordings'],1):
        if (task['question_id'],task['voice'],task['condition']) in done:continue
        strength = task.get('strength',0.)
        interventions = [dict(layer=31,group='answer',strength=strength)] if strength else []
        print(f'Evaluate [{i}/{len(plan["recordings"])}] {task["question_id"]} {task["condition"]}',flush=True)
        with install_interventions(layers,{(31,'answer'):vector},interventions,{},512):
            response, encoding = generate(task,plan,model,tokenizer,features)
        row = dict(task,response=response,reference_answer=task['answer'],prompt=plan['prompt'],seed=42,
            max_new_tokens=512,steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
            interventions=interventions,audio_encoding=encoding,vector_sha256=vector_hash if strength else None)
        with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(response,flush=True)


def report(plan, folder):
    from rescore_audio_steering import paired
    rows = read_rows(folder/'results.jsonl')
    if not rows:print(f'{folder}: no answers yet.');return
    scorer = NumericScorer(folder/'harness/upstream')
    groups = {}; scored = []
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
    lengths={}
    for row in rows:
        lengths.setdefault(row['condition'],[]).append(len(tokenizer.encode(row['response'],add_special_tokens=False)))
        metrics,predictions = scorer.score(row['response'],row['reference_answer'])
        groups.setdefault(row['condition'],{})[row['question_id'],row['voice']] = metrics
        scored.append(dict(row,scores=metrics,predictions=predictions))
    summary = []
    for condition, answers in groups.items():
        n = len(answers)
        print(f'{condition}: N={n}; '+ '; '.join(f'{metric}={sum(v[metric] for v in answers.values())/n:.1%}' for metric in next(iter(answers.values()))))
        print(f'  Response tokens excluding special tokens: mean={mean(lengths[condition]):.1f}; median={median(lengths[condition]):.1f}')
        if condition == 'neutral':continue
        keys = sorted(groups.get('neutral',{}).keys() & answers.keys())
        for metric in next(iter(answers.values())) if keys else []:
            result = paired([(groups['neutral'][k][metric],answers[k][metric]) for k in keys])
            summary.append(dict(condition=condition,metric=metric,**result))
            print(f'  {metric}: N={len(keys)} gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (folder/'summary.json').write_text(json.dumps(dict(scorer=scorer.provenance,paired=summary,lengths={c:dict(n=len(v),mean_tokens=mean(v),median_tokens=median(v)) for c,v in lengths.items()},note='Unadjusted exploratory paired tests/intervals.'),indent=2)+'\n')
    (folder/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emotion',choices=['happy','sad'],default='happy')
    parser.add_argument('--mode',choices=['run','prepare','summary','analysis'],default='run')
    parser.add_argument('--benchmark',type=Path,default=Path('outputs/reasoning-transfer/gsm8k'))
    parser.add_argument('--audio-sources',type=Path,nargs='+',default=AUDIO_SOURCES)
    parser.add_argument('--strengths',type=float,nargs='+',default=[1.])
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(); args.output = args.output or Path(f'outputs/diffa-gsm8k-{args.emotion}')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('summary','analysis'):
        plan=json.loads((args.output/'run_config.json').read_text())
        if args.mode=='analysis':analysis(plan,args.output);return
    else:
        plan=None
    tag='-'.join(f'{a:g}' for a in dict.fromkeys(args.strengths) if a)
    folder=args.output/'evaluations'/f'L31-answer-a{tag}';folder.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':report(plan,folder);return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args);evaluation=evaluation_plan(plan,args.strengths,args.output)
        freeze(folder/'run_config.json',evaluation)
        reuse(evaluation,folder,[args.benchmark,*args.audio_sources,*sorted((args.output/'evaluations').glob('*'))])
        if args.mode=='prepare':return
        NumericScorer(folder/'harness/upstream')  # Check evaluator before GPU work.
        done={(r['question_id'],r['voice'],r['condition']) for r in read_rows(folder/'results.jsonl')}
        if (args.output/'style-vectors.pt').exists() and all((r['question_id'],r['voice'],r['condition']) in done for r in evaluation['recordings']):
            report(plan,folder);return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared DIFFA GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            cache=Path(os.environ.get('HF_HOME','/workspace/.hf_home')).absolute()
            os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
            os.environ.pop('TRANSFORMERS_CACHE',None)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            import torch, transformers
            (args.output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),model_config=model.config.to_dict(),collector_sha256=digest(Path(__file__)),generation_helper_sha256=digest(Path(__file__).with_name('extract_vectors.py'))),indent=2,default=str)+'\n')
            collect(plan,args.output,model,tokenizer,features)
            evaluate(evaluation,folder,args.output,model,tokenizer,features)
        report(plan,folder)


if __name__=='__main__':main()
