"""Read-only counts of questions recognized as angry, happy AND sad; no inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

MODEL='emotion2vec/emotion2vec_plus_large'
EMOTIONS=('angry','happy','sad')
PREFIXES={'gsm8k':'gsm8k_test_','svamp':'svamp_chal-','multiarith':'multiarith_',
          'bbh3':'bbh_logical_three_','bbh5':'bbh_logical_five_','ifeval':'ifeval_'}
TARGETS={'gsm8k':425,'svamp':500,'multiarith':290,'bbh3':125,'bbh5':125,'ifeval':254}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count(root,cohorts=None):
    labels={}
    for path in sorted(root.rglob('results-emotion2vec.jsonl')):
        for row in read(path):
            if (row.get('model')==MODEL and row.get('processing_mode')=='whole_recording'
                    and row.get('classification_status')=='classified'):
                key=row.get('audio_sha256') or row.get('clip_id')
                if key:labels[key]=row['emotion']
    clips={}
    for path in sorted(root.rglob('audio_inventory.jsonl')):
        for row in read(path):
            key=row.get('audio_sha256') or row.get('clip_id')
            if not key or not row.get('aliases'):continue
            item=clips.setdefault(key,dict(paths=set(),aliases=[]))
            item['paths'].update(row.get('paths',[]))
            if row.get('audio_file'):item['paths'].add(row['audio_file'])
            item['aliases'].extend(row['aliases'])
    # Neutral recognition is irrelevant, but a surviving neutral recording is needed.
    available={n:{s:set() for s in ('neutral',*EMOTIONS)} for n in PREFIXES}
    recognized={n:{s:set() for s in EMOTIONS} for n in PREFIXES}
    pending=defaultdict(set)
    for key,clip in clips.items():
        if not any(Path(p).is_file() for p in clip['paths']):continue
        for alias in clip['aliases']:
            q=alias.get('question_id');style=alias.get('requested_style')
            if not isinstance(q,str) or style not in ('neutral',*EMOTIONS) or alias.get('voice')!='en-US-JennyNeural':continue
            name=next((n for n,prefix in PREFIXES.items() if q.startswith(prefix)),None)
            if name is None:continue
            available[name][style].add(q)
            if style=='neutral':continue
            if key not in labels:pending[name].add(key)
            elif labels[key]==style:recognized[name][style].add(q)
    result={}
    for name in PREFIXES:
        shared=available[name]['neutral'].intersection(*(recognized[name][s] for s in EMOTIONS))
        info=dict(target_collection=TARGETS[name],neutral_available=len(available[name]['neutral']),
                  available={s:len(available[name][s]) for s in EMOTIONS},
                  recognized={s:len(recognized[name][s]) for s in EMOTIONS},
                  shared_count=len(shared),shared_question_ids=sorted(shared),unclassified_emotional_clips=len(pending[name]))
        if cohorts and name in cohorts:
            data=cohorts[name];old_test=set(data['test_ids'])
            info['shared_outside_original_test']=len(shared-old_test)
            info['shared_in_original_collection']=len(shared & set(data['collection_ids']))
        result[name]=info
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs'))
    p.add_argument('--cohorts',type=Path,default=Path('outputs/diffa-angry-agreed/audit/cohorts.json'))
    p.add_argument('--json',type=Path,help='Optionally save counts and shared question IDs; default prints only.')
    args=p.parse_args();cohorts=json.loads(args.cohorts.read_text())['datasets'] if args.cohorts.exists() else None
    rows=count(args.root,cohorts)
    print('Dataset       Target   Angry OK   Happy OK   Sad OK   Shared3   Outside old test   Pending labels')
    for name,r in rows.items():
        outside=str(r['shared_outside_original_test']) if 'shared_outside_original_test' in r else 'n/a'
        print(f'{name:13} {r["target_collection"]:6} {r["recognized"]["angry"]:10} {r["recognized"]["happy"]:10} '
              f'{r["recognized"]["sad"]:8} {r["shared_count"]:9} {outside:>18} {r["unclassified_emotional_clips"]:16}')
        missing=[s for s in EMOTIONS if not r['available'][s]]
        if missing:print('  No available recordings for: '+', '.join(missing))
    print('Shared3 requires recognized angry/happy/sad on the same question, plus available neutral audio; neutral label is NOT filtered.')
    print('Counts use all surviving labeled Jenny clips, with any recognized rendition per style/question. Pending labels can increase the count.')
    print('Shared3 is the current maximum with a new split; Outside old test is the count available while preserving the original test IDs.')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
