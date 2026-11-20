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
from steer import read_rows


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
    variant=getattr(args,'prompt','source')
    frozen_step=variant=='calculations' and previous.get('model_key')=='step'
    ordered=list(pairs)
    if not frozen_step:random.Random(42).shuffle(ordered)
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
    if variant=='calculations':
        from step_calculations20 import PROMPT
        plan.update(protocol='step-gsm-calculations-v1',prompt=PROMPT,prompt_variant='calculations-required',
            prompt_selection='Same20 diagnostic inspected before this prompt extension; exploratory follow-up, not untouched confirmation.')
        if frozen_step:
            plan['selection']=f'First {count} IDs in original frozen Step baseline order; same audio/cohort, no reselection by outcomes.'
    args.output.mkdir(parents=True,exist_ok=True);freeze(args.output/'run_config.json',plan)
    print(f'Step GSM8K: {count} neutral/angry pairs, {2*count} answers; existing audio, no TTS or steering.',flush=True)
    print('Task prompt: '+plan['prompt'],flush=True)
    return plan


def reuse_diagnostic(plan,output,diagnostic):
    """Reuse only actual audio generations under the exact calculations recipe."""
    path=diagnostic/'run_config.json'
    if plan['prompt_variant']!='calculations-required' or not path.exists():return 0
    previous=json.loads(path.read_text())
    if (previous['protocol']!='step-calculations-same20-v1' or previous['prompt']!=plan['prompt']
            or previous['system_prompt']!=plan['system_prompt'] or previous['recipe']!=plan['recipe']
            or previous['seed']!=42 or previous['max_new_tokens']!=512):
        raise ValueError('Calculation diagnostic recipe differs; refusing mixed-prompt reuse.')
    tasks={(r['question_id'],r['voice'],r['style']):r for r in plan['recordings']}
    done={(r['question_id'],r['voice'],r['style']) for r in read_rows(output/'results.jsonl',repair=True)}
    copied=0
    for row in read_rows(diagnostic/'results.jsonl'):
        condition=row['condition']
        if condition not in ('neutral-calculations','angry-calculations'):continue
        style=condition.removesuffix('-calculations');key=row['question_id'],row['voice'],style
        if key not in tasks or key in done:continue
        task=tasks[key]
        if (row['prompt']!=plan['prompt'] or row.get('interventions')!=[] or row.get('seed')!=42
                or row.get('max_new_tokens')!=512 or row['model']!=plan['recipe']['model']
                or row['revision']!=plan['recipe']['revision'] or row['audio_sha256']!=task['audio_sha256']
                or row['reference_answer']!=task['reference_answer'] or row['style']!=style):
            raise ValueError('Diagnostic response/audio differs from the new frozen run.')
        saved=dict(row,condition=style,style=style,reused_from=str(diagnostic.absolute()),original_condition=condition)
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')
        done.add(key);copied+=1
    print(f'Reused {copied} matching neutral/angry diagnostic answers; no text-only answers copied.',flush=True)
    return copied


def evaluate(args,plan):
    with (args.output/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reuse_diagnostic(plan,args.output,getattr(args,'reuse_diagnostic',Path('outputs/step-calculations20')))
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
    parser.add_argument('--prompt',choices=('source','calculations'),default='source')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--reuse-diagnostic',type=Path,default=Path('outputs/step-calculations20'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--num-questions',type=int,default=400,help='0 uses the whole existing cohort; changing count requires a separate output.')
    args=parser.parse_args()
    args.source=args.source or Path('outputs/step-gsm400/run_config.json' if args.prompt=='calculations' else 'outputs/reasoning-transfer/gsm8k/run_config.json')
    args.output=args.output or Path('outputs/step-gsm400-calculations' if args.prompt=='calculations' else 'outputs/step-gsm400')
    if args.num_questions<0:parser.error('Question count must be nonnegative.')
    if args.mode=='summary':gsm.report(args.output)
    else:
        plan=prepare(args)
        if args.mode=='evaluate':evaluate(args,plan);gsm.report(args.output)
