"""Add Guy happy/sad on the same290 MultiArith questions; retain original neutral/angry generations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import run_reasoning_transfer as reasoning
from steer import read_rows


def prepare(args):
    config=args.baseline/'run_config.json';old=json.loads(config.read_text())
    if (old['task']!='multiarith' or old['voice']!='en-US-GuyNeural' or old['decoder']!='fast_dllm'
            or old['seed']!=42 or old['max_new_tokens']!=512 or old.get('steps',128)!=128
            or old.get('block_length',32)!=32 or old.get('temperature',0)!=0):raise ValueError('Need original Guy DIFFA recipe.')
    ids=list(dict.fromkeys(r['question_id'] for r in old['recordings']))
    if len(ids)!=290:raise ValueError('Expected original290 questions.')
    expected={(r['question_id'],r['voice'],r['condition']):r for r in old['recordings']}
    rows=read_rows(args.baseline/'results.jsonl');saved={}
    for row in rows:
        key=row['question_id'],row['voice'],row['condition']
        if (key not in expected or key in saved or row['condition'] not in ('neutral','angry')
                or row['prompt']!=old['prompt'] or row['seed']!=42 or row['max_new_tokens']!=512
                or row['audio_sha256']!=expected[key]['audio_sha256'] or row['interventions']):raise ValueError('Original Guy answer differs.')
        saved[key]=row
    if len(expected)!=580 or len(saved)!=580:raise ValueError('Finish all580 original Guy neutral/angry answers first.')
    args.source.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    questions=[old['bank'][qid] for qid in ids];reasoning.bank_file(args.source/'questions.jsonl',questions)
    subprocess.run([sys.executable,'-u','cloud_tts_smoke.py','--provider','azure','--azure-voice',old['voice'],
        '--azure-intensity','1.5','--input-jsonl',str(args.source/'questions.jsonl'),'--num-examples','290',
        '--styles','happy','sad','--output',str(args.source),'--resume','--retry-passes','5'],check=True)
    audio=reasoning.source_records(args.source,old['bank'],['happy','sad'])
    records=old['recordings']+[dict(audio[qid,style],condition=style) for qid in ids for style in ('happy','sad')]
    plan=dict(old,protocol='diffa-multiarith290-guy-four-emotions-v1',recordings=records,
        baseline_config_sha256=reasoning.digest(config),baseline_results_sha256=reasoning.digest(args.baseline/'results.jsonl'),
        baseline_source=str(args.baseline.absolute()),report_label='Unsteered Guy neutral/angry/happy/sad deliveries',
        selection='Same290 original test questions; neutral/angry saved answers reused verbatim; happy/sad added after inspecting '
            'neutral/angry results. Exploratory emotion comparisons, no outcome/classifier filtering or steering.')
    reasoning.freeze(args.output/'run_config.json',plan)
    present={}
    for row in read_rows(args.output/'results.jsonl',repair=True):
        key=row['question_id'],row['voice'],row['condition']
        if key in present:raise ValueError('Duplicate answer in four-emotion output.')
        present[key]=row
    with (args.output/'results.jsonl').open('a') as stream:
        for key,row in saved.items():
            if key in present:
                if present[key]['response']!=row['response'] or present[key]['audio_sha256']!=row['audio_sha256']:
                    raise ValueError('Reused baseline changed.')
                continue
            stream.write(json.dumps(dict(row,reused_from=str(args.baseline.absolute())))+'\n')
    print('Ready:580 neutral/angry answers reused; generate only580 happy/sad answers. Same prompt, seed42 and native decoder.',flush=True)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('outputs/multiarith290-guy-delivery'))
    p.add_argument('--source',type=Path,default=Path('outputs/azure-multiarith290-guy-happy-sad'))
    p.add_argument('--output',type=Path,default=Path('outputs/multiarith290-guy-four-emotions'))
    args=p.parse_args();prepare(args)


if __name__=='__main__':main()
