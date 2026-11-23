"""Pack/run an existing DIFFA vanilla vector at another layer on a second machine."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import random
import stat
import subprocess
import time
import zipfile

import diffa_gsm_vanilla as vanilla
from diffa_model_receipts import MODELS, cached_model_revisions, model_revisions
from steer import read_rows

MARKER = 'DIFFA_LAYER_PROBE_MANIFEST.json'
PACKAGES = ('torch', 'transformers', 'peft')


def encoded(value):
    return (json.dumps(value, indent=2)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package(source, destination, repo, layer=28, strength=1.):
    """Use the SAME target questions, raw vector and baseline answers as L31."""
    winners = vanilla.task.winners
    original = json.loads((source/'strength-1-unfitted/run_config.json').read_text())
    if original['target']['prompt'] != vanilla.PROMPT:
        raise ValueError('Expected the current vanilla GSM run, with no reasoning instruction.')
    emotion = original['emotion']
    vector_path = Path(original['source']['vector'])
    vector_bytes = vector_path.read_bytes()
    if sha(vector_bytes) != original['source']['vector_sha256']:
        raise ValueError('Vector differs from the running L31 experiment.')
    _, info = winners.read_vector(vector_path, emotion, layer=layer, group='answer')
    if set(info['extraction_ids']) & {r['question_id'] for r in original['target']['records']}:
        raise ValueError('Source fitting IDs overlap target questions.')
    relative = destination.relative_to(repo)
    job = deepcopy(original)
    job.update(protocol='diffa-vanilla-existing-vector-layer-probe-v1', layer=layer, group='answer', strengths=[0., strength],
               parent_l31_config_sha256=winners.digest(source/'strength-1-unfitted/run_config.json'))
    for key in ('source', 'target'):
        job[key]['vector'] = str(destination/'style-vectors.pt')
        job[key]['vector_sha256'] = sha(vector_bytes)
    raw = read_rows(source.parent.parent/'delivery/results.jsonl')
    baseline = {}
    expected = {(r['question_id'], r['voice']): r for r in job['target']['records']}
    for row in raw:
        key = row['question_id'], row['voice']
        if row['condition'] != 'neutral' or key not in expected:
            continue
        if (key in baseline or row.get('strength') != 0. or row.get('interventions') != []
                or not winners.reusable(row, {}, expected[key], job['target'])):
            raise ValueError('Neutral baseline differs from the exact source prompt/audio/decoding recipe.')
        baseline[key] = dict(row, vector_sha256=None, reused_from=str(source.parent.parent/'delivery'))
    if set(baseline) != set(expected):
        raise ValueError('Complete natural neutral baselines before making the layer bundle.')
    revisions = model_revisions() or cached_model_revisions(Path(os.environ.get('HF_HOME', '/workspace/.hf_home'))/'hub')
    if set(revisions) != set(MODELS):
        raise ValueError('Need all three original model revision SHAs before migrating the probe.')
    baseline_bytes = ''.join(json.dumps(baseline[r['question_id'], r['voice']])+'\n' for r in job['target']['records']).encode()
    files = {str(relative/'run_config.json'): encoded(job), str(relative/'style-vectors.pt'): vector_bytes,
             str(relative/'results.jsonl'): baseline_bytes, 'outputs/diffa-model-revisions.json': encoded(dict(models=revisions))}
    manifest = dict(repo=str(repo), folder=str(relative), models=revisions,
                    packages={p: version(p) for p in PACKAGES}, files={n: sha(data) for n, data in files.items()},
                    note='Existing DIFFA vector and exact L31 target/baseline cohort. No Phi weights or vectors; no recollection; no audio included.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(str(destination)+'.zip')
    if archive.exists():
        raise FileExistsError(f'Bundle already exists: {archive}; reuse it or select another --output.')
    temporary = archive.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
            for name, data in files.items():
                stream.writestr(name, data)
            stream.writestr(MARKER, encoded(manifest))
        with zipfile.ZipFile(temporary) as stream:
            if stream.testzip():
                raise ValueError('ZIP verification failed.')
        temporary.rename(archive)
    finally:
        temporary.unlink(missing_ok=True)
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive.read_bytes())}  {archive.name}\n')
    print(f'DOWNLOAD THIS SMALL FILE: {archive}\nSIZE: {archive.stat().st_size/2**20:.2f} MiB\n'
          f'{emotion}/{original["regime"]}, L{layer} answer x{strength:g}, {len(expected)} identical held-out questions. No audio/model weights.', flush=True)
    return archive


def unpack(archive, repo):
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read(MARKER))
        if manifest['repo'] != str(repo):
            raise ValueError(f'Clone must be at {manifest["repo"]}, matching the frozen audio paths.')
        for name, expected in manifest['files'].items():
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(stream.getinfo(name).external_attr >> 16):
                raise ValueError('Unsafe bundle member.')
            data = stream.read(name)
            if sha(data) != expected:
                raise ValueError(f'Corrupt bundle member: {name}')
            path = repo.joinpath(*member.parts)
            if not path.resolve().is_relative_to(repo):
                raise ValueError('Bundle path escapes repository.')
            if path.exists():
                existing = path.read_bytes()
                if existing == data or (name.endswith('/results.jsonl') and existing.startswith(data)):
                    continue
                if name == 'outputs/diffa-model-revisions.json' and json.loads(existing)['models'] == manifest['models']:
                    continue
                raise ValueError(f'Existing file differs; left unchanged: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.probe-restore-tmp')
            try:
                with temporary.open('xb') as output:
                    output.write(data)
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        folder = repo/manifest['folder']
        vanilla.task.winners.freeze(folder/'bundle-receipt.json', manifest)
    return folder


def validate(job, folder, check_audio=False):
    winners = vanilla.task.winners
    if winners.digest(job['source']['vector']) != job['source']['vector_sha256']:
        raise ValueError('Probe vector hash changed.')
    _, info = winners.read_vector(job['source']['vector'], job['emotion'], layer=job['layer'], group=job['group'])
    targets = {(r['question_id'], r['voice']): r for r in job['target']['records']}
    if set(info['extraction_ids']) & {q for q, _ in targets}:
        raise ValueError('Probe includes fitting questions.')
    if check_audio:
        for record in targets.values():
            path = Path(record['audio_file'])
            if not path.is_file() or winners.digest(path) != record['audio_sha256']:
                raise ValueError(f'Restore the same Jenny ZIP: missing or changed {path}')
    done = {}
    for row in read_rows(folder/'results.jsonl', repair=check_audio):
        key = row['question_id'], row['voice'], row['condition']
        record = targets.get(key[:2])
        strength = row.get('strength')
        interventions = [dict(layer=job['layer'], group=job['group'], strength=strength)] if strength else []
        if (record is None or key in done or strength not in job['strengths']
                or row['condition'] != winners.condition(strength, job['layer'], job['group'])
                or row.get('interventions') != interventions
                or row.get('vector_sha256') != (job['source']['vector_sha256'] if strength else None)
                or not winners.reusable(row, {}, record, job['target'])):
            raise ValueError('Probe answer cache differs from its frozen settings.')
        done[key] = row
    if sum(r['strength'] == 0. for r in done.values()) != len(targets):
        raise ValueError('Probe is missing source neutral baselines.')
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pack', 'run', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/diffa-gsm-vanilla/angry/random'))
    parser.add_argument('--layer', type=int, default=28)
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-count', type=int, default=3, help='Check source neutral reproducibility before steering.')
    args = parser.parse_args()
    if args.layer not in range(32) or args.strength == 0. or args.replay_count < 0:
        parser.error('Layer must be0..31, strength nonzero, replay count nonnegative.')
    repo = Path(__file__).resolve().parent
    name = f'gsm8k-{args.source.parent.name}-{args.source.name}-L{args.layer}-answer-a{args.strength:g}'
    folder = (args.output or Path('outputs/diffa-layer-probes')/name).resolve()
    if args.mode == 'pack':
        package(args.source.resolve(), folder, repo, args.layer, args.strength)
        return
    if args.bundle:
        folder = unpack(args.bundle, repo)
    job = json.loads((folder/'run_config.json').read_text())
    if args.mode == 'summary':
        validate(job, folder)
        print(f'DIFFA {job["emotion"]}/{job["regime"]}: L{job["layer"]} {job["group"]}', flush=True)
        vanilla.task.report_steering(folder)
        return
    with (folder/'run.lock').open('a') as own, Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = validate(job, folder, check_audio=True)
        pending = len(job['target']['records'])*len(job['strengths'])-len(done)
        if not pending:
            vanilla.task.report_steering(folder)
            return
        receipt = json.loads((folder/'bundle-receipt.json').read_text())
        for package_name in PACKAGES:
            if version(package_name) != receipt['packages'][package_name]:
                raise ValueError(f'{package_name} differs from source: {version(package_name)} vs {receipt["packages"][package_name]}. Use the source DIFFA environment.')
        if model_revisions() != receipt['models']:
            raise ValueError('Model revision receipts differ from source.')
        print(f'DIFFA model (hosted on this machine): L{job["layer"]} answer; {pending} generations pending. No recollection.', flush=True)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            time.sleep(30)
        from extract_vectors import load_model
        model, tokenizer, features = load_model()
        vanilla.task.winners.atomic_text(folder/'runtime.json', json.dumps(dict(
            runner_sha256=vanilla.task.winners.digest(__file__),
            generator_sha256=vanilla.task.winners.digest(vanilla.task.winners.__file__),
            models=model_revisions(), packages={p: version(p) for p in PACKAGES},
            model_config=model.config.to_dict()), indent=2, default=str)+'\n')
        replay = folder/'baseline-replay.json'
        if not replay.exists():
            sample = random.Random(42).sample(job['target']['records'], min(args.replay_count, len(job['target']['records'])))
            checks = []
            for record in sample:
                response, _ = vanilla.task.generate(record, job['target'], job['target']['prompt'], model, tokenizer, features)
                old = done[record['question_id'], record['voice'], 'neutral']['response']
                checks.append(dict(question_id=record['question_id'], identical=response == old, source=old, replay=response))
            vanilla.task.winners.atomic_text(replay, json.dumps(checks, indent=2)+'\n')
        checks = json.loads(replay.read_text())
        if any(not row['identical'] for row in checks):
            raise ValueError('Neutral replay differs on this machine. Inspect baseline-replay.json before comparing layers; no steering started.')
        vanilla.task.winners.generate_job(job, folder, model, tokenizer, features)
        validate(job, folder)
        vanilla.task.report_steering(folder, tokenizer)


if __name__ == '__main__':
    main()
