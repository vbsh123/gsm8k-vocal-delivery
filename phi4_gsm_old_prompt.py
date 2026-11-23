"""Fresh GSM audio with the older Phi reasoning instruction; held-out delivery-only prompt ablation."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess

from jenny_bundle_registry import sha
from steer import read_rows

PROTOCOL='phi4-gsm-fresh-audio-old-prompt-deliveries-v1'


def prepare(source,output):
    from evaluate_gsm_candidates import ORIGINAL
    from winning_diffa_sweeps import freeze
    parent=json.loads(source.read_text())
    train=[p['question_id'] for p in parent['collection']]
    ids=[p['question_id'] for p in parent['test']]
    if parent['task']!='gsm8k' or len(train)!=425 or len(ids)!=718 or len(set(train+ids))!=1143:
        raise ValueError('Need the exact original DIFFA-matched GSM425/718 source plan.')
    expected=parent['prompt'].replace('Give concise step-by-step reasoning with only the necessary calculations.',
                                      "Let's think step by step.")
    if expected!=ORIGINAL or parent['prompt']==ORIGINAL:
        raise ValueError('Unexpected source prompt; this ablation changes only the reasoning sentence.')
    plan=dict(protocol=PROTOCOL,task='gsm8k',source_config=str(source.absolute()),source_config_sha256=sha(source),
              recipe=parent['recipe'],seed=parent['seed'],max_new_tokens=parent['max_new_tokens'],prompt=ORIGINAL,
              excluded_collection_ids=train,test_ids=ids,bank={q:parent['bank'][q] for q in ids},
              recordings=[dict(p[style],condition=style,split='test') for p in parent['test'] for style in ('neutral','angry')],
              interventions=[],strength=0.,note='Fresh audio and held-out IDs identical to concise-prompt run. '
              'Older prompt only; generate both neutral/angry, no vector fitting or steering. No GPU queue wait.')
    output.mkdir(parents=True,exist_ok=True)
    freeze(output/'run_config.json',plan)
    print(f'Frozen {len(ids)} held-out questions, {2*len(ids)} neutral/angry answers; no collection or steering.',flush=True)
    return plan


def answers(plan,folder,repair=False):
    expected={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']}
    found={}
    for row in read_rows(folder/'results.jsonl',repair=repair):
        key=row['question_id'],row['voice'],row['condition']
        if key not in expected or key in found or row.get('run_config_sha256')!=sha(folder/'run_config.json'):
            raise ValueError('Duplicate or incompatible old-prompt answer.')
        required=dict(prompt=plan['prompt'],max_new_tokens=plan['max_new_tokens'],seed=plan['seed'],
                      model=plan['recipe']['model'],revision=plan['recipe']['revision'],
                      audio_sha256=expected[key]['audio_sha256'],reference_answer=expected[key]['reference_answer'],
                      interventions=[],strength=0.,split='test')
        if any(row.get(k)!=v for k,v in required.items()):
            raise ValueError('Saved old-prompt recording/model/generation settings changed.')
        found[key]=row
    return found


def report(plan,folder):
    from winning_diffa_sweeps import score_rows,atomic_text
    from rescore_audio_steering import paired
    from phi4_compare_delivery_cohorts import matched_deliveries,compare
    rows=list(answers(plan,folder).values())
    print(f'Older-prompt GSM delivery: {len(rows)}/{len(plan["recordings"])} answers saved.',flush=True)
    if not rows:
        return
    def score(values):
        return score_rows(dict(target=dict(task='gsm8k',name='gsm8k',bank=plan['bank'])),values,folder)[0]
    old_prompt=matched_deliveries(rows,heldout=True)
    if not old_prompt:
        print('No completed neutral/angry pairs yet.');return
    scored=score(list(old_prompt.values()))
    groups={s:{r['question_id']:r for r in scored if r['condition']==s} for s in ('neutral','angry')}
    ids=sorted(set(groups['neutral'])&set(groups['angry']))
    results={m:paired([(groups['neutral'][q]['metrics'][m],groups['angry'][q]['metrics'][m]) for q in ids])
             for m in scored[0]['metrics']}
    print(f'OLDER PROMPT — angry versus neutral: {len(ids)}/718 complete held-out questions.')
    for m,r in results.items():
        print(f'{m:18} neutral={r["baseline_correct"]}/{len(ids)}, angry={r["steered_correct"]}/{len(ids)}; '
              f'gain={r["gain_pp"]:+.2f} pp; p={r["mcnemar_exact_p"]:.4g}; CI={r["ci95_gain_pp"]}')
    lengths={s:dict(mean=statistics.mean(groups[s][q]['response_tokens'] for q in ids),
                    median=statistics.median(groups[s][q]['response_tokens'] for q in ids)) for s in groups}
    for s,info in lengths.items():print(s,'mean/median tokens:',round(info['mean'],1),info['median'])
    atomic_text(folder/'summary.json',json.dumps(dict(protocol=PROTOCOL,paired=results,lengths=lengths,matched=len(ids)),indent=2)+'\n')
    source=Path(plan['source_config'])
    from phi4_zip_recheck import saved_rows
    parent=json.loads(source.read_text())
    concise=matched_deliveries(saved_rows(source.parent/'deliveries.jsonl',parent).values(),heldout=True)
    if set(concise)&set(old_prompt):
        comparison=compare(concise,old_prompt,score,paired)
        n=len(comparison['common_question_voice_ids'])
        if any(v['same_audio']!=n or v['same_budget']!=n for v in comparison['cross_run'].values()):
            raise ValueError('Prompt-only comparison requires identical audio and token budgets.')
        comparison['labels']={'old-common':'concise prompt','fresh-common':'older Phi prompt'}
        atomic_text(folder/'prompt-comparison.json',json.dumps(comparison,indent=2)+'\n')
        print(f'\nSAME AUDIO / SAME {n} QUESTIONS — angry-minus-neutral gain:')
        for m,r in comparison['cohorts']['old-common'].items():
            new=comparison['cohorts']['fresh-common'][m]
            print(f'{m:18} concise={r["gain_pp"]:+.2f} pp; older prompt={new["gain_pp"]:+.2f} pp')


def run(plan,folder,native):
    import phi4_all_layer_diagnostic as hooks
    done=answers(plan,folder,repair=True)
    with (folder/'results.jsonl').open('a') as stream:
        for index,record in enumerate(plan['recordings'],1):
            key=record['question_id'],record['voice'],record['condition']
            if key in done:continue
            print(f'Older prompt [{index}/{len(plan["recordings"])}] {record["question_id"]} {record["condition"]}',flush=True)
            trace=hooks.generate(native,record,plan['prompt'],max_new_tokens=plan['max_new_tokens'])
            row=dict(record,**trace,protocol=PROTOCOL,run_config_sha256=sha(folder/'run_config.json'),
                     prompt=plan['prompt'],seed=plan['seed'],max_new_tokens=plan['max_new_tokens'],
                     model=plan['recipe']['model'],revision=plan['recipe']['revision'],interventions=[],strength=0.)
            stream.write(json.dumps(row)+'\n');stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--source',type=Path,default=Path('outputs/phi4-diffa-cohorts-three-directions/gsm8k/run_config.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/phi4-gsm-fresh-audio-old-prompt'))
    args=parser.parse_args()
    if args.mode=='summary':
        report(json.loads((args.output/'run_config.json').read_text()),args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args.source,args.output)
        if args.mode=='prepare':return
        pending=len(answers(plan,args.output,repair=True))<len(plan['recordings'])
        if pending:
            from evaluate_bbh_candidates import concurrent_budget_now
            from steer_phi4_gsm_reverse import NativePhi
            concurrent_budget_now(15.)
            native=NativePhi(11)
            run(plan,args.output,native)
        subprocess.run([str(Path('.venv-diffa/bin/python').absolute()),'-u',__file__,'--mode','summary',
                        '--output',str(args.output)],check=True)


if __name__=='__main__':main()
