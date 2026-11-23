"""Pack/run an existing DIFFA vanilla vector at another layer on a second machine."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib.metadata import version
import io
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


def fork_layers(archive, repo, layers, destination=None):
    """Change only the layers, preserving the exact bundle cohort/vector/baselines."""
    winners = vanilla.task.winners
    with zipfile.ZipFile(archive) as stream:
        reference = json.loads(stream.read(MARKER))
        if reference['repo'] != str(repo):
            raise ValueError('Reference bundle repository path differs from this machine.')
        prefix = reference['folder']
        source = {}
        for filename in ('run_config.json', 'style-vectors.pt', 'results.jsonl'):
            member = prefix+'/'+filename
            data = stream.read(member)
            if sha(data) != reference['files'][member]:
                raise ValueError(f'Corrupt reference bundle file: {member}')
            source[filename] = data
    job = json.loads(source['run_config.json'])
    if job['emotion'] not in vanilla.EMOTIONS or job['regime'] not in vanilla.REGIMES:
        raise ValueError('Unknown source emotion/regime.')
    if not layers or any(not isinstance(layer, int) or layer not in range(32) for layer in layers):
        raise ValueError('Need one or more layers0..31.')
    layers = sorted(set(layers))
    if destination is None and reference.get('layers') == layers:
        print(f'Reusing existing bundle with layers={layers}: {archive}', flush=True)
        return Path(archive)
    for layer in layers:
        _, info = winners.read_vector(io.BytesIO(source['style-vectors.pt']), job['emotion'], layer=layer, group=job['group'])
    if set(info['extraction_ids']) & {r['question_id'] for r in job['target']['records']}:
        raise ValueError('Reference includes fitting questions.')
    count = len(job['target']['records'])
    strengths = [a for a in job['strengths'] if a != 0.]
    if len(strengths) != 1:
        raise ValueError('Reference must contain one steering strength.')
    name = f'gsm8k-{job["emotion"]}-{job["regime"]}-L{min(layers)}-{max(layers)}-{job["group"]}-a{strengths[0]:g}-n{count}'
    destination = destination or repo/'outputs/diffa-layer-probes'/name
    relative = destination.relative_to(repo)
    job = deepcopy(job)
    job.update(layer=layers[0], reference_bundle_sha256=sha(Path(archive).read_bytes()))
    for key in ('source', 'target'):
        job[key]['vector'] = str(destination/'style-vectors.pt')
    files = {str(relative/'run_config.json'): encoded(job), str(relative/'style-vectors.pt'): source['style-vectors.pt'],
             str(relative/'results.jsonl'): source['results.jsonl'],
             'outputs/diffa-model-revisions.json': encoded(dict(models=reference['models']))}
    manifest = dict(reference, folder=str(relative), layers=layers, files={n: sha(data) for n, data in files.items()},
                    reference_bundle_sha256=sha(Path(archive).read_bytes()),
                    note='Exact existing vector, question IDs and baseline responses preserved; only the tested layers change. No resampling.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = Path(str(destination)+'.zip')
    if output.exists():
        with zipfile.ZipFile(output) as stream:
            if json.loads(stream.read(MARKER)) != manifest or any(sha(stream.read(n)) != sha(data) for n, data in files.items()):
                raise ValueError(f'Existing layer-sweep bundle differs: {output}')
    else:
        temporary = output.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
                for member, data in files.items():
                    stream.writestr(member, data)
                stream.writestr(MARKER, encoded(manifest))
            with zipfile.ZipFile(temporary) as stream:
                if stream.testzip():
                    raise ValueError('Layer bundle ZIP verification failed.')
            temporary.rename(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Layer sweep: {job["emotion"]}/{job["regime"]}, layers={layers}, x{strengths[0]:g}, '
          f'EXACT SAME {count} questions. Bundle: {output}', flush=True)
    return output


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


def report_natural_benefit(job, baseline_rows, data_root, output):
    """Natural emotional audio vs neutral on exactly the probe's selected questions."""
    winners = vanilla.task.winners
    if not (data_root/'split.json').exists() or not (data_root/'delivery/results.jsonl').exists():
        raise FileNotFoundError(f'Natural emotion answers missing under {data_root}. Run this report on the original DIFFA machine.')
    manifest = json.loads((data_root/'split.json').read_text())
    if vanilla.fingerprint(manifest) != job['split_sha256']:
        raise ValueError('Natural answers belong to a different frozen prompt/audio cohort.')
    emotion = job['emotion']
    expected = {(r['question_id'], r['voice']): r for r in job['target']['records']}
    neutral_snapshot = {(r['question_id'], r['voice']): r for r in baseline_rows if r['strength'] == 0.}
    if set(neutral_snapshot) != set(expected):
        raise ValueError('Bundle baselines do not match its target questions.')
    selected = {}
    for row in read_rows(data_root/'delivery/results.jsonl'):
        key = row['question_id'], row['voice']
        style = row['condition']
        if key not in expected or style not in ('neutral', emotion):
            continue
        record = dict(expected[key], audio_sha256=manifest['audio_sha256'][key[0]][style])
        if ((key, style) in selected or row.get('strength') != 0. or row.get('interventions') != []
                or not winners.reusable(row, {}, record, job['target'])):
            raise ValueError('Natural response differs from the probe prompt, audio hash or decoding recipe.')
        if style == 'neutral' and row['response'] != neutral_snapshot[key]['response']:
            raise ValueError('Natural neutral answer differs from the exact baseline used by the layer sweep.')
        selected[key, style] = row
    ids = [r['question_id'] for r in job['target']['records']]
    plan = dict(name=job['target']['name'], emotion=emotion, prompt=job['target']['prompt'],
                test_ids=ids, target=job['target'])
    output.mkdir(parents=True, exist_ok=True)
    winners.atomic_text(output/'selection.json', json.dumps(dict(question_ids=ids, split_sha256=job['split_sha256'],
        note='Exact layer-probe subset. Natural emotion audio versus the same neutral baselines. No e2v or correctness filtering.'), indent=2)+'\n')
    print(f'NATURAL {emotion} AUDIO vs NEUTRAL: exact{len(ids)} layer-sweep questions; no steering; no extra filtering.', flush=True)
    vanilla.task.report_delivery(plan, output, rows=list(selected.values()), cohort='exact layer-sweep subset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pack', 'run', 'summary', 'deliveries'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/diffa-gsm-vanilla/angry/random'))
    parser.add_argument('--layer', type=int, default=28)
    parser.add_argument('--layers', type=int, nargs='+', help='Pack a sweep, or select new layers with --bundle while preserving its exact questions.')
    parser.add_argument('--questions', type=int, help='Seed42 random subset of the existing unfitted test questions.')
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('outputs/diffa-gsm-vanilla'),
                        help='Saved natural responses for --mode deliveries; use on the original DIFFA machine.')
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
    if args.questions is not None and args.mode != 'pack':
        parser.error('--questions belongs to --mode pack. Run/summary preserve the existing bundle questions.')
    if args.layers and args.mode != 'pack' and (not args.bundle or args.mode not in ('run', 'summary')):
        parser.error('Use --layers with --mode pack, or with --bundle in run/summary to reuse exactly its questions.')
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
    if args.mode == 'deliveries':
        if not args.bundle:
            parser.error('--mode deliveries requires --bundle to identify the exact sweep questions.')
        with zipfile.ZipFile(args.bundle) as stream:
            manifest = json.loads(stream.read(MARKER))
            prefix = manifest['folder']
            job_bytes = stream.read(prefix+'/run_config.json')
            baseline_bytes = stream.read(prefix+'/results.jsonl')
            if sha(job_bytes) != manifest['files'][prefix+'/run_config.json'] or sha(baseline_bytes) != manifest['files'][prefix+'/results.jsonl']:
                raise ValueError('Bundle question/baseline metadata is corrupt.')
            job = json.loads(job_bytes)
            baseline_rows = [json.loads(line) for line in baseline_bytes.splitlines() if line.strip()]
        report_natural_benefit(job, baseline_rows, args.data_root,
            repo/'outputs/diffa-layer-natural-benefit'/Path(prefix).name)
        return
    if args.bundle:
        print(f'Reading probe bundle: {args.bundle}', flush=True)
        if args.layers:
            args.bundle = fork_layers(args.bundle, repo, args.layers, destination=args.output.resolve() if args.output else None)
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
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another copy of this probe is already running: {folder}. This duplicate launch is exiting.') from None
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
        while True:
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print('Waiting for the existing DIFFA/Phi GPU queue to finish; retry in30 seconds.', flush=True)
                time.sleep(30)
        while True:
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            active = [pid.strip() for pid in processes.splitlines() if pid.strip() and pid.strip() != str(os.getpid())]
            if not active:
                break
            print(f'GPU still active: PIDs={active}; waiting30 seconds before model load.', flush=True)
            time.sleep(30)
        print('Loading DIFFA and pinned model weights; missing files will download.', flush=True)
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
