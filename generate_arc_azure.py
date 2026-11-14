"""Generate 500 fixed text-only ARC-Challenge test questions in Azure Jenny deliveries."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from urllib.request import urlopen

from generate_commonsense_azure import STYLES,duration_report

SOURCE_URL='https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet'
SOURCE_SHA256='62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9'
VISUAL=re.compile(r'\b(?:diagram|figure|graph|picture|illustration)\b|\b(?:table (?:below|above)|following table)\b|<img|\\(?:frac|sqrt)|[_^]\{',re.I)


def select_questions(rows,count):
    selected=[];excluded=[]
    for index,row in enumerate(rows):
        labels=list(row['choices']['label']);texts=list(row['choices']['text'])
        reason='not_four_choices' if len(labels)!=4 or len(texts)!=4 else None
        if VISUAL.search(row['question']+' '+' '.join(texts)):reason='visual_reference_or_math_markup'
        if reason:
            excluded.append(dict(source_id=row['id'],reason=reason));continue
        if len(set(labels))!=4 or row['answerKey'] not in labels:raise ValueError('Unexpected ARC choice labels')
        choices=[dict(label=letter,text=text) for letter,text in zip('ABCD',texts)]
        answer='ABCD'[labels.index(row['answerKey'])]
        spoken=row['question']+'\n'+'\n'.join(f'Option {c["label"]}: {c["text"]}.' for c in choices)
        selected.append(dict(question_id='arc_challenge_test_'+row['id'],question=spoken,
            question_stem=row['question'],choices=choices,answer=answer,answer_text=texts[labels.index(row['answerKey'])],
            source_dataset='ARC-Challenge',source_split='test',source_id=row['id'],source_index=index,
            original_choice_labels=labels,original_answer=row['answerKey'],source_url=SOURCE_URL,source_sha256=SOURCE_SHA256))
    random.Random(42).shuffle(selected)
    if not 1<=count<=len(selected):raise ValueError(f'Only {len(selected)} eligible ARC questions')
    return selected[:count],excluded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/azure-arc-challenge500'))
    p.add_argument('--num-examples',type=int,default=500)
    p.add_argument('--styles',nargs='+',choices=STYLES,default=STYLES)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'generation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=args.output/'arc-challenge-test.parquet'
        if not source.exists():
            print('Download official ARC-Challenge test parquet (about 204 kB).',flush=True)
            with urlopen(SOURCE_URL,timeout=120) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256:raise ValueError('ARC source checksum changed')
            temporary=source.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(source)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:raise ValueError('Cached ARC checksum differs')
        import pyarrow.parquet as pq
        bank,excluded=select_questions(pq.read_table(source).to_pylist(),args.num_examples)
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bank)
        path=args.output/'questions.jsonl'
        if path.exists() and path.read_text()!=text:raise ValueError('Selected questions changed; use a different folder')
        path.write_text(text)
        (args.output/'selection.json').write_text(json.dumps(dict(seed=42,source_sha256=SOURCE_SHA256,
            excluded=excluded,selected_questions=len(bank),choice_order='Original order; labels remapped to ABCD'),indent=2)+'\n')
        print(f'{len(bank)} fixed questions × {len(args.styles)} deliveries. Selection never uses model answers.',flush=True)
        command=[sys.executable,'-u','cloud_tts_smoke.py','--provider','azure','--azure-voice','en-US-JennyNeural',
            '--azure-intensity','1.5','--input-jsonl',str(path),'--num-examples',str(len(bank)),
            '--styles',*args.styles,'--output',str(args.output),'--resume','--retry-passes','5']
        if args.prepare_only:command.append('--prepare-only')
        subprocess.run(command,check=True)
        if not args.prepare_only:duration_report(args.output)


if __name__=='__main__':main()
