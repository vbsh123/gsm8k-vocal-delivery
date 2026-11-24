"""Move the original BBH arithmetic pilot audio to Phi and evaluate the same spoken questions."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import zipfile

import phi4_more_bare_eval as phi
import run_arithmetic_audio_pilot as arithmetic
from phi4_sweep import job_lock
from winning_diffa_sweeps import atomic_text, freeze

CONFIG = 'diffa-run_config.json'
SOURCE = 'bbh-source.json'
PROMPTS = {'supply': arithmetic.REASONING_PROMPT, 'bare': arithmetic.BARE_PROMPT}


def pilot_records(config_data, source_data):
    parent = json.loads(config_data)
    bank = {q['question_id']: q for q in arithmetic.source_questions(source_data)}
    if (parent['protocol'] != arithmetic.PROTOCOL or parent['task'] != arithmetic.TASK
            or parent['bank'] != bank or parent['question_order'] != list(bank)
            or parent['prompts'] != arithmetic.PROMPTS or parent['max_new_tokens'] != 512):
        raise ValueError('Expected the original DIFFA 50-question BBH arithmetic pilot.')
    records = {(r['question_id'], r['condition']):r for r in parent['records']}
    if len(records) != len(parent['records']) or len(records) != 4*len(bank):
        raise ValueError('Expected four unique DIFFA conditions per question.')
    clips = []
    for q, item in bank.items():
        for style in ('angry','neutral'):
            row = records[q,style+'-reasoning']
            if (row['voice'] != arithmetic.base.VOICE or row['style'] != style
                    or row['reference_answer'] != item['answer']
                    or records[q,style]['audio_sha256'] != row['audio_sha256']):
                raise ValueError(f'Inconsistent DIFFA audio or target for {q}/{style}.')
            clips.append((f'audio/{q}-{style}.wav', row))
    return bank, clips


def make_bundle(path, source, pilot):
    config = (pilot/'run_config.json').read_bytes()
    data = (source/SOURCE).read_bytes()
    bank, clips = pilot_records(config, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp)/'audio.zip'
        with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(CONFIG, config)
            archive.writestr(SOURCE, data)
            for member, row in clips:
                wav = Path(row['audio_file'])
                if not wav.is_file():
                    wav = source/wav.name
                content = wav.read_bytes()
                if hashlib.sha256(content).hexdigest() != row['audio_sha256']:
                    raise ValueError(f'Original audio differs: {wav}')
                archive.writestr(member, content)
        staged.replace(path)
    print(f'Bundle ready: {path.resolve()} ({len(bank)} questions, {len(clips)} verified WAVs).')


def prepare(bundle, output, labels):
    with zipfile.ZipFile(bundle) as archive:
        config = archive.read(CONFIG)
        bank, clips = pilot_records(config, archive.read(SOURCE))
        source_hash = hashlib.sha256(config).hexdigest()
        for label in PROMPTS:
            frozen = output/label/'run_config.json'
            if frozen.exists() and json.loads(frozen.read_text())['diffa_config_sha256'] != source_hash:
                raise ValueError(f'Pilot configuration differs from the existing run in {output}.')
        recordings = []
        (output/'audio').mkdir(parents=True, exist_ok=True)
        for member, row in clips:
            wav = (output/member).resolve()
            content = archive.read(member)
            if hashlib.sha256(content).hexdigest() != row['audio_sha256']:
                raise ValueError(f'Bundle audio hash differs: {member}')
            if not wav.exists() or phi.sha(wav) != row['audio_sha256']:
                pending = wav.with_suffix('.tmp')
                pending.write_bytes(content)
                pending.replace(wav)
            recordings.append(dict(row, audio_file=str(wav), condition=row['style']))
    for label in labels:
        folder = output/label
        plan = dict(protocol='phi4-bbh-arithmetic50-v1', task=arithmetic.TASK, model='phi',
                    recipe=phi.RECIPE, prompt_label=label, prompt=PROMPTS[label], question_order=list(bank),
                    bank=bank, recordings=recordings, max_new_tokens=512, seed=42, interventions=[], strength=0.,
                    diffa_config_sha256=source_hash)
        phi.validate(plan)
        freeze(folder/'run_config.json', plan)
        atomic_text(folder/'prompt.txt', plan['prompt']+'\n')
    arithmetic.scoring_preflight(output)
    print(f'Prepared Phi for {len(bank)} questions and {len(recordings)} verified WAVs. Prompts: {", ".join(labels)}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bundle','prepare','run','summary'))
    parser.add_argument('--bundle', type=Path, default=Path('/workspace/bbh-arithmetic50-audio.zip'))
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-arithmetic50'))
    parser.add_argument('--pilot', type=Path, default=Path('outputs/diffa-bbh-arithmetic50'))
    parser.add_argument('--output', type=Path, default=Path('outputs/phi4-bbh-arithmetic50'))
    parser.add_argument('--prompts', choices=PROMPTS, nargs='+', default=['supply'])
    args = parser.parse_args()
    if args.mode == 'bundle':
        make_bundle(args.bundle, args.source, args.pilot)
        return
    if args.mode == 'prepare':
        with job_lock(args.output):
            prepare(args.bundle, args.output, args.prompts)
        return
    jobs = []
    for label in dict.fromkeys(args.prompts):
        folder = args.output/label
        config = folder/'run_config.json'
        if args.mode == 'summary' and not config.exists():
            print(f'Phi arithmetic {label}: awaiting preparation.')
            continue
        plan = json.loads(config.read_text())
        if plan['prompt'] != PROMPTS[label] or plan['prompt_label'] != label:
            raise ValueError('Run directory contains a different prompt.')
        if args.mode == 'summary':
            phi.report(folder, plan)
        else:
            jobs.append((folder, plan))
    if args.mode == 'run':
        with job_lock(args.output):
            pending = [(f,p) for f,p in jobs if len(phi.answers(f,p,repair=True)) < len(p['recordings'])]
            if pending:
                with phi.model_session() as native:
                    for folder, plan in pending:
                        freeze(folder/'runtime.json', dict(packages={k:version(k) for k in ('torch','transformers')},
                               model=phi.RECIPE, model_config=native.model.config.to_dict()))
                        phi.generate(folder, plan, native)


if __name__ == '__main__':
    main()
