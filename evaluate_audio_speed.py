"""DIFFA accuracy on original Spoken GSM at pitch-preserving playback speeds.

Same questions at every rate, no cropping, and seed 42 before every generation.
1x bypasses time stretching. Saved answers resume; settings/cohort are frozen.
"""

import argparse
import json
from math import comb, isfinite
from pathlib import Path
import random
import time

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows


RATES = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]


def condition(rate):
    return "baseline" if rate==1 else f"speed-{rate:g}x"


def select_recordings(records, root, count, rates, sf):
    selected=[]
    seen=set()
    for r in records:
        if r['style']!='original' or r['clip_id'] in seen:
            continue
        path=(root/r['audio_path']).resolve()
        info=sf.info(path)
        if not 0<info.frames/info.samplerate<=30*min(rates):
            continue
        selected.append({**r,'audio_path':str(path),'audio_bytes':path.stat().st_size,
                         'audio_mtime_ns':path.stat().st_mtime_ns})
        seen.add(r['clip_id'])
        if len(selected)==count:
            break
    if len(selected)!=count:
        raise ValueError(f'Need {count} original clips <= {30*min(rates):g}s; found {len(selected)}')
    return selected


def reusable_answers(source, plan):
    if not (source/'run_config.json').exists():
        print('No previous speed run found; generating the required answers.',flush=True)
        return []
    previous=json.loads((source/'run_config.json').read_text())
    keys=('seed','prompt','steps','block_length','max_new_tokens','temperature','decoder',
          'sample_rate','transform','librosa','max_seconds')
    if any(previous['settings'].get(k)!=plan['settings'][k] for k in keys):
        print('Previous speed recipe differs; no answers reused.',flush=True)
        return []
    old={r['clip_id']:r for r in previous['recordings']}
    selected={r['clip_id']:r for r in plan['recordings']}
    copied=[]
    for row in read_rows(source/'results.jsonl'):
        cid, rate=row['clip_id'],row['rate']
        if cid not in selected or old.get(cid)!=selected[cid] or rate not in plan['rates']:
            continue
        if (row.get('seed')!=42 or row.get('prompt')!=PROMPT or row['condition']!=condition(rate)
                or str(row['reference_answer'])!=str(selected[cid]['answer'])):
            continue
        copied.append(dict(row,reused_from=str(source.resolve())))
    return copied


def report(output):
    plan=json.loads((output/"run_config.json").read_text())
    rows={(r["clip_id"],r["condition"]):r for r in read_rows(output/"results.jsonl")}
    baseline={clip:r for (clip,c),r in rows.items() if c=="baseline"}
    print(f"Original-speed baseline: {sum(r['correct'] for r in baseline.values())}/{len(baseline)} correct.")
    print(f"{'Speed':>6} {'Pairs':>6} {'Accuracy':>10} {'Baseline':>10} {'Wins':>6} {'Losses':>6} {'Exact p':>9}")
    summary=[]
    for rate in plan["rates"]:
        pairs=[(b,rows[clip,condition(rate)]) for clip,b in baseline.items() if (clip,condition(rate)) in rows]
        if not pairs:
            continue
        n=len(pairs)
        k=sum(r["correct"] for b,r in pairs)
        kb=sum(b["correct"] for b,r in pairs)
        w=sum(r["correct"] and not b["correct"] for b,r in pairs)
        l=sum(b["correct"] and not r["correct"] for b,r in pairs)
        d=w+l
        p=min(1,2*sum(comb(d,i) for i in range(min(w,l)+1))/2**d)
        summary.append(dict(rate=rate,pairs=n,correct=k,baseline_correct=kb,wins=w,losses=l,mcnemar_exact_p=p))
        print(f"{rate:>5g}x {n:>6} {k/n:>10.1%} {kb/n:>10.1%} {w:>6} {l:>6} {p:>9.4f}")
    print("Same-question comparisons against 1x. Parse failures count as incorrect; p-values are unadjusted.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path",type=Path,default=Path("outputs/spoken-gsm-original"))
    p.add_argument("--num-examples",type=int,default=100)
    p.add_argument("--output",type=Path,default=Path("outputs/spoken-gsm-speed100"))
    p.add_argument("--rates",type=float,nargs='+',default=RATES,help='Playback rates; 1x baseline is always included.')
    p.add_argument("--reuse-from",type=Path,help='Copy compatible completed answers for overlapping questions and rates.')
    p.add_argument("--prepare-only",action="store_true",help="Select/freeze questions without loading DIFFA.")
    p.add_argument("--summary-only",action="store_true")
    p.add_argument("--save-audio",action="store_true",help="Save the 16-kHz waveforms used for inference, for listening.")
    args=p.parse_args()
    if args.summary_only:
        report(args.output)
        return
    import librosa
    import soundfile as sf

    if args.num_examples<1 or any(not isfinite(r) or r<=0 for r in args.rates):
        p.error('Need a positive question count and finite positive playback rates')
    rates=sorted(set([1.0,*args.rates]))
    records=read_rows(args.path/"manifest.jsonl")
    selected=select_recordings(records,args.path,args.num_examples,rates,sf)
    conditions=[dict(name="baseline")]+[dict(name=condition(r)) for r in rates if r!=1]
    plan=dict(recordings=selected,conditions=conditions,rates=rates,
              settings=dict(seed=42,prompt=PROMPT,steps=128,block_length=32,max_new_tokens=512,
                            temperature=0.0,decoder="fast_dllm",sample_rate=16000,
                            transform="librosa.effects.time_stretch; rate1 bypassed",
                            librosa=librosa.__version__,max_seconds=30,
                            selection=f"first N original recordings <={30*min(rates):g}s; no correctness or emotion filtering"))
    args.output.mkdir(parents=True,exist_ok=True)
    config=args.output/"run_config.json"
    if config.exists() and json.loads(config.read_text())!=plan:
        p.error("Source recordings/settings changed; use another --output")
    results_path=args.output/"results.jsonl"
    if results_path.exists() and not config.exists():
        p.error("Results exist without frozen settings; use another --output")
    rows=read_rows(results_path,repair=not args.prepare_only)
    allowed={(r["clip_id"],c["name"]) for r in selected for c in conditions}
    if any((r["clip_id"],r["condition"]) not in allowed or r.get("seed")!=42 for r in rows):
        p.error("Saved answers have a different cohort, condition or seed")
    config.write_text(json.dumps(plan,indent=2)+"\n")
    completed={(r["clip_id"],r["condition"]) for r in rows}
    if args.reuse_from and not args.prepare_only:
        reused=0
        with results_path.open('a') as stream:
            for row in reusable_answers(args.reuse_from,plan):
                key=(row['clip_id'],row['condition'])
                if key in completed:
                    continue
                stream.write(json.dumps(row)+'\n')
                completed.add(key)
                reused+=1
        print(f'Reused {reused} compatible saved answers; original results unchanged.',flush=True)
    pending=len(allowed)-len(completed)
    print(f"{len(selected)} matched questions x {len(rates)} speeds; {pending} generations pending.",flush=True)
    print("Rates:",rates,"; seed always 42; pitch preserved; no truncation.",flush=True)
    if args.prepare_only:
        return
    if not pending:
        report(args.output)
        return
    import numpy as np
    import torch
    from extract_vectors import load_model,gen
    model,tokenizer,feature_extractor=load_model()
    for i,r in enumerate(selected,1):
        if all((r["clip_id"],c["name"]) in completed for c in conditions):
            continue
        audio,sr=sf.read(r["audio_path"],dtype="float32")
        if audio.ndim==2:
            audio=audio.mean(axis=1)
        if sr!=16000:
            audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        for rate in [1.0]+[v for v in rates if v!=1]:
            name=condition(rate)
            if (r["clip_id"],name) in completed:
                continue
            changed=audio if rate==1 else librosa.effects.time_stretch(audio,rate=rate)
            if len(changed)>30*16000:
                raise ValueError(f'{r["clip_id"]} at {rate:g}x exceeds 30 seconds; no audio was cropped')
            if args.save_audio:
                folder=args.output/"audio"
                folder.mkdir(exist_ok=True)
                sf.write(folder/f"{r['clip_id']}__{rate:g}x.wav",changed,16000,subtype="FLOAT")
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            print(f"[{i}/{len(selected)}] {r['clip_id']} {rate:g}x ({len(changed)/16000:.2f}s)",flush=True)
            start=time.monotonic()
            response=gen(changed,16000,model,tokenizer,feature_extractor,PROMPT,512)
            row=dict(clip_id=r["clip_id"],question_id=r["question_id"],condition=name,rate=rate,
                     seed=42,prompt=PROMPT,response=response,duration_seconds=len(changed)/16000,
                     generation_seconds=time.monotonic()-start,**score_response(response,r["answer"]))
            with results_path.open("a") as stream:
                stream.write(json.dumps(row)+"\n")
                stream.flush()
            completed.add((r["clip_id"],name))
            print(f"  correct={row['correct']} answer={row['predicted_answer']}",flush=True)
    report(args.output)


if __name__=="__main__":
    main()
