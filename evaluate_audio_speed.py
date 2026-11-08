"""DIFFA accuracy on original Spoken GSM at nine pitch-preserving playback speeds.

Same questions at every rate, no cropping, and seed 42 before every generation.
1x bypasses time stretching. Saved answers resume; settings/cohort are frozen.
"""

import argparse
import json
from math import comb
from pathlib import Path
import random
import time

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows


RATES = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]


def condition(rate):
    return "baseline" if rate==1 else f"speed-{rate:g}x"


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
    p.add_argument("--prepare-only",action="store_true",help="Select/freeze questions without loading DIFFA.")
    p.add_argument("--summary-only",action="store_true")
    p.add_argument("--save-audio",action="store_true",help="Save the 16-kHz waveforms used for inference, for listening.")
    args=p.parse_args()
    if args.summary_only:
        report(args.output)
        return
    if args.num_examples<1:
        p.error("--num-examples must be positive")
    import soundfile as sf
    import librosa
    records=read_rows(args.path/"manifest.jsonl")
    selected=[]
    seen=set()
    for r in records:
        if r["style"]!="original":
            continue
        path=(args.path/r["audio_path"]).resolve()
        info=sf.info(path)
        # At 0.7x, an original <=21 seconds remains within the shared 30s limit.
        if not 0<info.frames/info.samplerate<=30*min(RATES):
            continue
        if r["question_id"] in seen:
            p.error(f"Duplicate source question: {r['question_id']}")
        seen.add(r["question_id"])
        selected.append({**r,"audio_path":str(path),"audio_bytes":path.stat().st_size,
                         "audio_mtime_ns":path.stat().st_mtime_ns})
        if len(selected)==args.num_examples:
            break
    if len(selected)!=args.num_examples:
        p.error(f"Only {len(selected)} originals <=21s available; prepare more originals or lower --num-examples")
    conditions=[dict(name="baseline")]+[dict(name=condition(r)) for r in RATES if r!=1]
    plan=dict(recordings=selected,conditions=conditions,rates=RATES,
              settings=dict(seed=42,prompt=PROMPT,steps=128,block_length=32,max_new_tokens=512,
                            temperature=0.0,decoder="fast_dllm",sample_rate=16000,
                            transform="librosa.effects.time_stretch; rate1 bypassed",
                            librosa=librosa.__version__,max_seconds=30,
                            selection="first N original recordings <=21s; no correctness or emotion filtering"))
    args.output.mkdir(parents=True,exist_ok=True)
    config=args.output/"run_config.json"
    if config.exists() and json.loads(config.read_text())!=plan:
        p.error("Source recordings/settings changed; use another --output")
    results_path=args.output/"results.jsonl"
    if results_path.exists() and not config.exists():
        p.error("Results exist without frozen settings; use another --output")
    rows=read_rows(results_path)
    allowed={(r["clip_id"],c["name"]) for r in selected for c in conditions}
    if any((r["clip_id"],r["condition"]) not in allowed or r.get("seed")!=42 for r in rows):
        p.error("Saved answers have a different cohort, condition or seed")
    config.write_text(json.dumps(plan,indent=2)+"\n")
    completed={(r["clip_id"],r["condition"]) for r in rows}
    pending=len(allowed)-len(completed)
    print(f"{len(selected)} matched questions x {len(RATES)} speeds; {pending} generations pending.",flush=True)
    print("Rates:",RATES,"; seed always 42; pitch preserved; no truncation.",flush=True)
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
        for rate in [1.0]+[v for v in RATES if v!=1]:
            name=condition(rate)
            if (r["clip_id"],name) in completed:
                continue
            changed=audio if rate==1 else librosa.effects.time_stretch(audio,rate=rate)
            if len(changed)>30*16000 or not np.isfinite(changed).all():
                raise ValueError(f"Invalid or >30s transformed audio: {r['clip_id']} {rate}x")
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
