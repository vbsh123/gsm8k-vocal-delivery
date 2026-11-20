"""Same20 GSM8K questions: Step text / neutral audio / angry audio, required calculations.

Standalone diagnostic using the existing Step environment/adapter. No TTS or new weights.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
from statistics import mean,median
import subprocess
import time

from bbh_candidate_adapters import checkpoint,MODELS
from kimi_bbh_layer_diagnostic import digest,freeze
from steer import read_rows
from step_audio_adapter import SYSTEM,native_inputs,decode_suffix

PROMPT=('Solve the math problem provided below. Show your calculations before giving the final answer. '
        'Write the intermediate arithmetic or equations and explain how they lead to the result. '
        'Do not give only the final answer. After the calculations, end with exactly one line: #### <number>.\n\n')


def prepare(source,output):
    previous=json.loads((source/'run_config.json').read_text())
    if previous['model_key']!='step' or previous['recipe']['model']!=MODELS['step']['model'] or previous['recipe']['revision']!=MODELS['step']['revision']:
        raise ValueError('Expected the pinned Step GSM400 baseline.')
    ids=list(previous['bank'])[:20]
    if len(ids)!=20:raise ValueError('Need20 distinct baseline questions.')
    selected={q:previous['bank'][q] for q in ids}
    audio={(r['question_id'],r['condition']):r for r in previous['recordings']}
    tasks=[]
    for q in ids:
        tasks.append(dict(question_id=q,condition='text-calculations',question=selected[q]['question'],reference_answer=str(selected[q]['answer']).rsplit('####',1)[-1].strip()))
        for style in ('neutral','angry'):
            row=audio[q,style]
            if digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Original audio changed.')
            tasks.append(dict(row,condition=style+'-calculations',reference_answer=str(selected[q]['answer']).rsplit('####',1)[-1].strip()))
    plan=dict(protocol='step-calculations-same20-v1',source=str(source.absolute()),source_sha256=digest(source/'run_config.json'),
        recipe=previous['recipe'],system_prompt=SYSTEM,prompt=PROMPT,bank=selected,recordings=tasks,
        selection='First20 question IDs in the frozen Step400 bank order; no selection by outcomes.',
        seed=42,max_new_tokens=512,note='Exploratory text/audio and prompt diagnostic; no causal reasoning claim.')
    freeze(output/'run_config.json',plan)
    original=[dict(r,condition=r['condition']+'-original') for r in read_rows(source/'results.jsonl') if r['question_id'] in selected]
    (output/'original-results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in original))
    return plan


def run(plan,output):
    import numpy as np
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    done={(r['question_id'],r['condition']) for r in read_rows(output/'results.jsonl',repair=True)}
    pending=[r for r in plan['recordings'] if (r['question_id'],r['condition']) not in done]
    if not pending:return
    with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
        print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(gpu,fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work.',flush=True);time.sleep(30)
        if transformers.__version__!=MODELS['step']['transformers']:raise ValueError('Use the existing DIFFA/Step environment.')
        path=checkpoint('step')
        tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,padding_side='right')
        model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16).eval().cuda()
        import sys
        features=sys.modules[type(model).__module__]
        tokenizer.eos_token='<|EOT|>';eos=tokenizer.convert_tokens_to_ids('<|EOT|>');model.config.eos_token_id=eos
        config=GenerationConfig(max_new_tokens=512,pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True)
        (output/'runtime.json').write_text(json.dumps(dict(checkpoint=path,model_class=type(model).__name__,torch=torch.__version__,transformers=transformers.__version__,eos_token_id=eos,
            model_code_sha256=digest(features.__file__),adapter_sha256=digest(Path(__file__).with_name('step_audio_adapter.py'))),indent=2)+'\n')
        for i,row in enumerate(pending,1):
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            if row['condition']=='text-calculations':
                # Same instruction as audio, followed by question text only, never reference/solution.
                parts=[f'<|BOT|>system\n{SYSTEM}<|EOT|>','<|BOT|>human\n',PROMPT,row['question'],'<|EOT|>','<|BOT|>assistant\n']
                ids=torch.cat([tokenizer(text=p,return_tensors='pt',padding=True)['input_ids'] for p in parts],dim=-1).to(model.device)
                inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids))
            else:
                if digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Audio changed.')
                inputs=native_inputs(tokenizer,features,row['audio_file'],PROMPT,model.device)
            if inputs['input_ids'].shape[-1]+512>model.config.text_config.max_position_embeddings:raise ValueError('Context overflow; no cropping.')
            print(f'[{i}/{len(pending)}] {row["question_id"]} {row["condition"]}',flush=True)
            with torch.inference_mode():ids=model.generate(**inputs,generation_config=config,tokenizer=tokenizer)
            suffix=ids[0,inputs['input_ids'].shape[-1]:];response,tokens=decode_suffix(tokenizer,suffix,eos)
            saved=dict(row,response=response,response_tokens=tokens,text_tokens=len(tokenizer.encode(response,add_special_tokens=False)),generated_token_ids=suffix.tolist(),
                prompt=PROMPT,seed=42,max_new_tokens=512,model=MODELS['step']['model'],revision=MODELS['step']['revision'],interventions=[],ended_with_eos=bool(len(suffix) and suffix[-1].item()==eos))
            with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')
            print(response,flush=True)


def report(output):
    from run_reasoning_transfer import NumericScorer
    from rescore_audio_steering import paired
    rows=read_rows(output/'original-results.jsonl')+read_rows(output/'results.jsonl')
    if not rows:print('No answers yet.');return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(checkpoint('step'),trust_remote_code=True)
    scorer=NumericScorer(output/'harness/upstream');groups={};scored=[]
    for r in rows:
        r=dict(r,text_tokens=len(tokenizer.encode(r['response'],add_special_tokens=False)))
        metrics,predictions=scorer.score(r['response'],r['reference_answer'])
        groups.setdefault(r['condition'],{})[r['question_id']]=dict(r,metrics=metrics,predictions=predictions)
        scored.append(dict(r,metrics=metrics,predictions=predictions))
    print('Condition                   N  Numeric Strict Format Mean tokens Median tokens')
    for condition,answers in groups.items():
        values=list(answers.values());n=len(values);lengths=[v['text_tokens'] for v in values]
        rates=[100*sum(v['metrics'][m] for v in values)/n for m in ('flexible-numeric','strict-match','format-compliant')]
        print(f'{condition:27} {n:2d} '+ ' '.join(f'{x:6.1f}' for x in rates)+f' {mean(lengths):10.1f} {median(lengths):13.1f}')
    results=[]
    for base,test in [('neutral-original','neutral-calculations'),('angry-original','angry-calculations'),('text-calculations','neutral-calculations'),('neutral-calculations','angry-calculations')]:
        a,b=groups.get(base,{}),groups.get(test,{});ids=sorted(a.keys()&b.keys())
        if not ids:continue
        result=paired([(a[q]['metrics']['flexible-numeric'],b[q]['metrics']['flexible-numeric']) for q in ids])
        results.append(dict(reference=base,test=test,**result));print(f'{test} vs {base}: N={len(ids)}; gain={result["gain_pp"]:+.1f} pp; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (output/'summary.json').write_text(json.dumps(dict(paired=results,scorer=scorer.provenance,note='20-question exploratory diagnostic; all reported text lengths exclude special tokens; raw suffix lengths/IDs remain saved separately.'),indent=2)+'\n')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['run','summary'],default='run')
    p.add_argument('--source',type=Path,default=Path('outputs/step-gsm400'))
    p.add_argument('--output',type=Path,default=Path('outputs/step-calculations20'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':report(args.output)
    else:
        with (args.output/'queue.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            plan=prepare(args.source,args.output);run(plan,args.output);report(args.output)
