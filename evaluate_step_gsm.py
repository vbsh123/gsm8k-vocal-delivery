"""Step neutral/angry on fixed existing DIFFA GSM8K pairs; no TTS or steering."""
import argparse
import fcntl
import json
from pathlib import Path
import random
import subprocess
import time

from bbh_candidate_adapters import MODELS
import evaluate_bbh_candidates as candidates
import evaluate_gsm_candidates as gsm
from kimi_bbh_layer_diagnostic import digest,freeze


def prepare(args):
    previous=json.loads(args.source.read_text())
    if previous['task']!='gsm8k':raise ValueError('Expected saved GSM8K evaluation configuration.')
    bank=previous['bank']
    pairs={}
    for row in previous['recordings']:
        if row['condition'] not in ('neutral','angry'):continue
        key=row['question_id'];style=row['condition']
        if style in pairs.setdefault(key,{}):raise ValueError('Duplicate unsteered source recording.')
        if row.get('style',style)!=style:raise ValueError('Source condition/style disagree.')
        pairs[key][style]=row
    ordered=list(pairs);random.Random(42).shuffle(ordered)
    count=args.num_questions or len(ordered)
    if count<=0 or count>len(ordered):raise ValueError('Requested question count exceeds existing cohort.')
    selected=ordered[:count];records=[]
    for qid in selected:
        pair=pairs[qid]
        if set(pair)!= {'neutral','angry'}:raise ValueError('Missing paired source delivery: '+qid)
        if pair['neutral']['voice']!=pair['angry']['voice']:raise ValueError('Paired voices differ.')
        gold=str(bank[qid]['answer']).rsplit('####',1)[-1].strip()
        for style in ('neutral','angry'):
            row=pair[style];path=Path(row['audio_file'])
            if digest(path)!=row['audio_sha256']:raise ValueError('Source audio changed: '+str(path))
            if str(row.get('reference_answer',row.get('answer'))).rsplit('####',1)[-1].strip()!=gold:
                raise ValueError('Source reference differs from bank.')
            # Copy only audio identity fields, never DIFFA decoder/steering metadata.
            records.append({**{k:row[k] for k in ('question_id','voice','audio_file','audio_sha256','seconds')},
                'style':style,'condition':style,'reference_answer':gold})
    plan=dict(protocol='step-gsm-fixed-delivery-v1',task='gsm8k',model_key='step',source=str(args.source.resolve()),
        source_config_sha256=digest(args.source),bank={qid:bank[qid] for qid in selected},recordings=records,
        prompt=previous['prompt'],prompt_variant='source-reasoning',seed=42,max_new_tokens=512,
        recipe=dict(MODELS['step'],dtype='bfloat16',do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True,text_only_output=True),
        interventions=[],system_prompt='You are a helpful assistant.',
        audio_encoding='Pinned native Step log-mel functions, full 16kHz audio in ordered 25-second windows; no transcription/crop.',
        selection=f'First {count} IDs after seed42 shuffle of saved DIFFA GSM8K cohort order; no outcome/emotion-classifier/duration selection.',
        scoring='Installed upstream GSM8K strict/flexible filters and exact match; numeric equivalence and valid format additional. '
            'Native audio-adapted inference, not official text-harness prompting. Paired p-values/intervals unadjusted.')
    args.output.mkdir(parents=True,exist_ok=True);freeze(args.output/'run_config.json',plan)
    print(f'Step GSM8K: {count} neutral/angry pairs, {2*count} answers; existing audio, no TTS or steering.',flush=True)
    print('Task prompt: '+plan['prompt'],flush=True)
    return plan


def evaluate(args,plan):
    with (args.output/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not candidates.pending([(args.output,plan)]):
            print('All answers complete; no GPU wait or model load.');return
        # Wait for the entire transfer queue, including CPU scoring between GPU jobs.
        queue=Path('outputs/winning-transfer/queue.lock')
        if queue.parent.exists():
            with queue.open('a') as transfer:
                print('Waiting for DIFFA transfer queue to finish.',flush=True)
                fcntl.flock(transfer,fcntl.LOCK_EX)
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            candidates.run([(args.output,plan)],'step')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','evaluate','summary'),default='evaluate')
    parser.add_argument('--source',type=Path,default=Path('outputs/reasoning-transfer/gsm8k/run_config.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/step-gsm400'))
    parser.add_argument('--num-questions',type=int,default=400,help='0 uses the whole existing cohort; changing count requires a separate output.')
    args=parser.parse_args()
    if args.num_questions<0:parser.error('Question count must be nonnegative.')
    if args.mode=='summary':gsm.report(args.output)
    else:
        plan=prepare(args)
        if args.mode=='evaluate':evaluate(args,plan);gsm.report(args.output)
