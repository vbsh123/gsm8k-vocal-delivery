"""Prepare the six-style Olivia cohort, or synthesize it through Azure/Vercel."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from xml.etree import ElementTree
import zipfile


HERE = Path(__file__).resolve().parent
STYLES = ('neutral', 'angry', 'happy', 'sad', 'disgusted', 'fearful')
VOICE = 'en-US-Olivia:MAI-Voice-2.1'
SOURCES = {
    'gsm8k': ['azure-jenny50', 'azure-jenny500-neutral', 'azure-jenny-next500', 'azure-jenny-remaining-test'],
    'multiarith': ['azure-multiarith580'],
    'svamp': ['azure-svamp500', 'azure-svamp-remaining500'],
    'bbh3': ['azure-bbh-logical250-steering'],
    'bbh5': ['azure-bbh-five250'],
    'ifeval': ['azure-spoken-ifeval'],
}


def plan_text(root, source):
    member = f'outputs/{source}/plan.jsonl'
    if root.is_file():
        with zipfile.ZipFile(root) as archive:
            return archive.read(member).decode()
    for path in (root/member, root/source/'plan.jsonl'):
        if path.is_file():
            return path.read_text()
    raise FileNotFoundError(f'Missing {member} under {root}. Pass --source pointing to the six-dataset Jenny ZIP or its extracted directory.')


def cohort(task, root):
    bank = {}
    for source in SOURCES[task]:
        for line in plan_text(root, source).splitlines():
            row = json.loads(line)
            if row['style'] != 'neutral':
                continue
            q = row['question_id']
            text = ''.join(ElementTree.fromstring(row['request']).itertext()).strip()
            item = dict(question_id=q, question=text, answer=row['answer'])
            if bank.setdefault(q, item) != item:
                raise ValueError(f'Conflicting Jenny text or answer for {q}')
    ids = sorted(bank)
    if task == 'multiarith':
        ids = sorted(json.loads((HERE/'data/multiarith_original_jenny290.json').read_text())['questions'])
    return [bank[q] for q in random.Random(42).sample(ids, 200)]


def save_once(path, data):
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f'Existing settings differ: {path}')
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def run_task(args, task):
    folder = args.output/task
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder/'questions.jsonl'
        if args.source is not None or not path.exists():
            places = (HERE/'outputs/jenny-zip-audio', HERE)
            root = args.source or next((p for p in places if (p/'outputs'/SOURCES[task][0]/'plan.jsonl').exists()), HERE)
            rows = cohort(task, root)
            save_once(path, ''.join(json.dumps(r)+'\n' for r in rows))
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 200 or len({r['question_id'] for r in rows}) != 200:
            raise ValueError(f'Invalid 200-question cohort at {path}')
        chars = sum(len(r['question']) for r in rows)*len(STYLES)
        print(f'{task}: {len(rows)} questions x {len(STYLES)} styles; {chars:,} spoken characters; approx. USD {chars*22/1e6:.2f}.', flush=True)
        if args.mode == 'prepare':
            return
        out = folder/args.mode
        out.mkdir(exist_ok=True)
        if args.mode == 'azure':
            env = dict(os.environ)
            for name in ('KEY', 'REGION'):
                env.setdefault('AZURE_SPEECH_'+name, env.get('SPEECH_'+name, ''))
            subprocess.run([sys.executable, str(HERE/'cloud_tts_smoke.py'), '--provider', 'azure',
                            '--input-jsonl', str(path), '--num-examples', '200', '--azure-voice', VOICE,
                            '--styles', *STYLES, '--output', str(out), '--resume'], check=True, env=env)
        else:
            plan = [dict(question_id=r['question_id'], question=r['question'], answer=r['answer'],
                         provider='vercel', model='microsoft/mai-voice-2.1', voice='en-US-Olivia',
                         style=s, audio_path=f'q{i:05d}-vercel-{s}.wav')
                    for i,r in enumerate(rows) for s in STYLES]
            save_once(out/'plan.jsonl', ''.join(json.dumps(r)+'\n' for r in plan))
            subprocess.run(['node', str(HERE/'mai_vercel.mjs'), str(out)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'azure', 'vercel'))
    parser.add_argument('datasets', nargs='+', choices=tuple(SOURCES)+('all',))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/mai-olivia200'))
    args = parser.parse_args()
    for task in (SOURCES if 'all' in args.datasets else dict.fromkeys(args.datasets)):
        run_task(args, task)
