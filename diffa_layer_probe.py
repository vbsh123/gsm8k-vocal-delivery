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


def original_job(source):
    path = source/'strength-1-unfitted/run_config.json'
    if path.exists():
        return json.loads(path.read_text()), vanilla.task.winners.digest(path)
    # Both vectors are finalized before L31 steering starts; agreed need not wait for random.
    collection = json.loads((source/'run_config.json').read_text())
    root = source.parent.parent
    plans = {e: {r: json.loads((root/e/r/'run_config.json').read_text()) for r in vanilla.REGIMES} for e in vanilla.EMOTIONS}
    scoped = vanilla.delivery_scope(plans, root)
    job = vanilla.steering_job(collection, source, scoped[collection['emotion']][collection['regime']])
    return job, sha(encoded(job))


def package(source, destination, repo, layer=28, strength=1., layers=None, questions=None):
    """Use the SAME target questions, raw vector and baseline answers as L31."""
    winners = vanilla.task.winners
    original, parent_hash = original_job(source)
    if original['target']['prompt'] != vanilla.PROMPT:
        raise ValueError('Expected the current vanilla GSM run, with no reasoning instruction.')
    emotion = original['emotion']
    vector_path = Path(original['source']['vector'])
    vector_bytes = vector_path.read_bytes()
    if sha(vector_bytes) != original['source']['vector_sha256']:
        raise ValueError('Vector differs from the running L31 experiment.')
    for selected_layer in layers or [layer]:
        _, info = winners.read_vector(vector_path, emotion, layer=selected_layer, group='answer')
    if set(info['extraction_ids']) & {r['question_id'] for r in original['target']['records']}:
        raise ValueError('Source fitting IDs overlap target questions.')
    relative = destination.relative_to(repo)
    job = deepcopy(original)
    job.update(protocol='diffa-vanilla-existing-vector-layer-probe-v1', layer=layer, group='answer', strengths=[0., strength],
               parent_l31_config_sha256=parent_hash)
    if questions is not None:
        population = sorted(job['target']['records'], key=lambda r: (r['question_id'], r['voice']))
        if not 1 <= questions <= len(population):
            raise ValueError(f'Need1..{len(population)} test questions.')
        sample = random.Random(42).sample(population, questions)
        keep = {(r['question_id'], r['voice']) for r in sample}
        job['target']['records'] = [r for r in job['target']['records'] if (r['question_id'], r['voice']) in keep]
        job['target']['bank'] = {r['question_id']: job['target']['bank'][r['question_id']] for r in job['target']['records']}
        job['subset'] = dict(seed=42, population=len(population), questions=questions,
                             rule='Uniform sample without replacement, independent of correctness; identical across layers.')
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
    if layers is not None:
        manifest['layers'] = list(layers)
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
          f'{emotion}/{original["regime"]}, layers={layers or [layer]} answer x{strength:g}, '
          f'{len(expected)} identical held-out questions per layer. No audio/model weights.', flush=True)
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


def resolve_audio(job, roots):
    """Relocate exact bytes from the Phi ZIP directory without changing frozen configs."""
    roots = list(dict.fromkeys(Path(root).resolve() for root in roots))
    alternatives = {}
    for root in roots:
        manifest = root/'JENNY_AUDIO_MANIFEST.json'
        if manifest.exists():
            for entry in json.loads(manifest.read_text()).get('files', []):
                if entry.get('audio_sha256') and entry.get('member'):
                    alternatives.setdefault(entry['audio_sha256'], []).append(root/entry['member'])
    found, mappings = {}, []
    for record in job['target']['records']:
        key = record['audio_sha256']
        original = record['audio_file']
        if key not in found:
            candidates = [Path(original)]
            if '/outputs/' in original:
                suffix = original.split('/outputs/', 1)[1]
                candidates.extend(root/'outputs'/suffix for root in roots)
            candidates.extend(alternatives.get(key, []))
            for path in dict.fromkeys(candidates):
                if path.is_file() and vanilla.task.winners.digest(path) == key:
                    found[key] = str(path.resolve())
                    break
            else:
                raise ValueError(f'Exact audio bytes not found for {record["question_id"]}; expected SHA256={key}. '
                                 f'Searched original path and {[str(r) for r in roots]}. '
                                 'Use --audio-root with the directory containing the extracted Jenny ZIP.')
        mappings.append(dict(question_id=record['question_id'], audio_sha256=key,
                             source_audio_file=original, runtime_audio_file=found[key]))
    return found, mappings


def with_audio_paths(job, paths):
    result = deepcopy(job)
    result['target']['records'] = [dict(record, audio_file=paths[record['audio_sha256']])
                                   for record in job['target']['records']]
    return result


def probe_jobs(folder, job, receipt):
    layers = receipt.get('layers')
    if layers is None:
        return [(folder, job)]
    baseline = ''.join(json.dumps(row)+'\n' for row in read_rows(folder/'results.jsonl') if row['strength'] == 0.)
    result = []
    for layer in layers:
        selected = dict(job, layer=layer)
        child = folder/f'L{layer}'
        child.mkdir(exist_ok=True)
        vanilla.task.winners.freeze(child/'run_config.json', selected)
        path = child/'results.jsonl'
        if not path.exists():
            vanilla.task.winners.atomic_text(path, baseline)
        validate(selected, child)
        result.append((child, selected))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pack', 'run', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/diffa-gsm-vanilla/angry/random'))
    parser.add_argument('--layer', type=int, default=28)
    parser.add_argument('--layers', type=int, nargs='+', help='Pack a sequential sweep using one shared vector file and question subset.')
    parser.add_argument('--questions', type=int, help='Seed42 random subset of the existing unfitted test questions.')
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--audio-root', action='append', type=Path,
                        help='Extracted Jenny ZIP root; default outputs/jenny-zip-audio and the repository.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-count', type=int, default=3, help='Check source neutral reproducibility before steering.')
    args = parser.parse_args()
    if args.layer not in range(32) or args.strength == 0. or args.replay_count < 0:
        parser.error('Layer must be0..31, strength nonzero, replay count nonnegative.')
    if args.layers:
        args.layers = sorted(set(args.layers))
        if any(layer not in range(32) for layer in args.layers):
            parser.error('Layers must be0..31.')
    if args.questions is not None and args.questions < 1:
        parser.error('--questions must be positive.')
    if (args.layers or args.questions is not None) and args.mode != 'pack':
        parser.error('--layers and --questions belong to --mode pack. Run/summary use the settings inside the bundle.')
    repo = Path(__file__).resolve().parent
    layer_name = f'L{args.layer}' if not args.layers else f'L{min(args.layers)}-{max(args.layers)}'
    name = f'gsm8k-{args.source.parent.name}-{args.source.name}-{layer_name}-answer-a{args.strength:g}'
    if args.questions is not None:
        name += f'-n{args.questions}'
    folder = (args.output or Path('outputs/diffa-layer-probes')/name).resolve()
    if args.mode == 'pack':
        package(args.source.resolve(), folder, repo, args.layers[0] if args.layers else args.layer, args.strength,
                layers=args.layers, questions=args.questions)
        return
    if args.bundle:
        folder = unpack(args.bundle, repo)
    job = json.loads((folder/'run_config.json').read_text())
    receipt = json.loads((folder/'bundle-receipt.json').read_text())
    cases = probe_jobs(folder, job, receipt)
    if args.mode == 'summary':
        for child, selected in cases:
            validate(selected, child)
            print(f'DIFFA {selected["emotion"]}/{selected["regime"]}: L{selected["layer"]} {selected["group"]}', flush=True)
            vanilla.task.report_steering(child)
        return
    with (folder/'run.lock').open('a') as own, Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots = [*(args.audio_root or [repo/'outputs/jenny-zip-audio']), repo]
        paths, mappings = resolve_audio(job, roots)
        vanilla.task.winners.atomic_text(folder/'audio-relocations.json', json.dumps(dict(
            source_config_sha256=vanilla.task.winners.digest(folder/'run_config.json'), recordings=mappings), indent=2)+'\n')
        changed = sum(row['source_audio_file'] != row['runtime_audio_file'] for row in mappings)
        print(f'Audio verified: {len(mappings)} exact hashes; {changed} paths relocated. Original recordings/configs unchanged.', flush=True)
        job = with_audio_paths(job, paths)
        cases = [(child, with_audio_paths(selected, paths)) for child, selected in cases]
        done = validate(job, folder, check_audio=True)
        pending = sum(len(selected['target']['records'])*len(selected['strengths'])-len(validate(selected, child))
                      for child, selected in cases)
        if not pending:
            for child, selected in cases:
                vanilla.task.report_steering(child)
            return
        for package_name in PACKAGES:
            if version(package_name) != receipt['packages'][package_name]:
                raise ValueError(f'{package_name} differs from source: {version(package_name)} vs {receipt["packages"][package_name]}. Use the source DIFFA environment.')
        if model_revisions() != receipt['models']:
            raise ValueError('Model revision receipts differ from source.')
        print(f'DIFFA {job["emotion"]}/{job["regime"]}: layers={[selected["layer"] for _,selected in cases]}; '
              f'{len(job["target"]["records"])} questions each; {pending} generations pending. One model load; no recollection.', flush=True)
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
        for child, selected in cases:
            print(f'STAGE: {selected["emotion"]}/{selected["regime"]} L{selected["layer"]} answer; '
                  f'{len(selected["target"]["records"])} identical target questions.', flush=True)
            vanilla.task.winners.generate_job(selected, child, model, tokenizer, features)
            validate(selected, child)
            vanilla.task.report_steering(child, tokenizer)


if __name__ == '__main__':
    main()
