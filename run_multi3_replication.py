"""Matched MultiArith three-ask evaluation on Phi or with spoken instructions on DIFFA."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from statistics import mean, median

from multiarith_instruction_checks import THIRD_ASK, checks
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from run_diffa_spoken_prompt import generate as generate_flipped, saved
from run_multiarith_reasoning_ablation import SENTENCE, STYLES, EXTRA_STYLES
from run_reasoning_transfer import generate_audio, source_records
import winning_diffa_sweeps as winners

DATA = Path(__file__).resolve().parent / 'data/multiarith_three_ask_pairs290.json'
AUDIO = Path('inputs/multi3-jenny290')


def resolve_phi_recordings(records, folder, workspace=Path('/workspace')):
    """Reuse exactly the clips from the original DIFFA comparison, wherever staged."""
    base = json.loads(DATA.with_name('multiarith_original_jenny290.json').read_text())['questions']
    extra = json.loads(DATA.with_name('multiarith_three_asks_audio.json').read_text())
    staged = Path('outputs/jenny-zip-audio')
    paths, missing = {}, {}
    for row in records:
        q, style, key = row['question_id'], row['style'], row['audio_sha256']
        original = (base if style in STYLES else extra)[q][style]
        if original['audio_sha256'] != key:
            raise ValueError(f'Original recording manifests disagree for {q}/{style}.')
        relative = Path(original['audio_file'])
        cached = folder/'restored-audio'/f'{key}.wav'
        for path in (Path(row['audio_file']), relative, staged/relative, cached):
            if path.is_file() and winners.digest(path) == key:
                paths[key] = path
                break
        else:
            missing[key] = (q, style)

    # Bundle manifests also record aliases when the original path was external.
    for root in (staged, Path('.')):
        manifest = root/'JENNY_AUDIO_MANIFEST.json'
        if missing and manifest.is_file():
            for entry in json.loads(manifest.read_text())['files']:
                key = entry['audio_sha256']
                path = root/entry['member']
                if key in missing and path.is_file() and winners.digest(path) == key:
                    paths[key] = path
                    del missing[key]

    # A worker may have the original transfer ZIP without an extracted copy.
    for archive in sorted(workspace.glob('jenny-six-datasets-audio-*.zip')):
        if not missing:
            break
        with zipfile.ZipFile(archive) as z:
            for entry in json.loads(z.read('JENNY_AUDIO_MANIFEST.json'))['files']:
                key = entry['audio_sha256']
                if key not in missing:
                    continue
                data = z.read(entry['member'])
                if hashlib.sha256(data).hexdigest() != key:
                    raise ValueError(f'Jenny archive hash mismatch in {archive}: {entry["member"]}')
                path = folder/'restored-audio'/f'{key}.wav'
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(data)
                tmp.replace(path)
                paths[key] = path
                del missing[key]
    if missing:
        examples = ', '.join(f'{q}/{s}' for q, s in list(missing.values())[:3])
        raise FileNotFoundError(f'{len(missing)} original Jenny recordings unavailable, including {examples}. '
                                'Restore inputs/ from multi3-phi290.zip in the repo and rerun.')
    return paths


def prepare(folder, experiment):
    data = json.loads(DATA.read_text())
    bank = data['questions']
    original = data['original_prompt']
    if len(bank) != 290 or original.count(SENTENCE) != 1:
        raise ValueError('Unexpected MultiArith cohort or original prompt.')
    prompt = original.replace(SENTENCE, SENTENCE + THIRD_ASK + ' ')
    spoken = prompt.replace('in the audio', 'in writing').replace('#### <number>', 'four hash signs, a space, then the number')
    styles = STYLES
    if experiment == 'phi':
        from phi4_gsm_bare_eval import RECIPE
        styles += EXTRA_STYLES
    else:
        from diffa_gsm_vanilla import RECIPE
        cue = dict(question_id='three_asks', question=spoken, answer='')
        audio = folder/'instruction-audio'
        if not (folder/'run_config.json').exists():
            generate_audio(audio, [cue], styles)
        clips = source_records(audio, {'three_asks': cue}, styles)

    records = []
    for q, item in bank.items():
        for style in styles:
            if experiment == 'phi':
                path = AUDIO/f'{q}-{style}.wav'
                audio_hash = item['audio_sha256'][style]
                source = dict(audio_file=str(path), audio_sha256=audio_hash)
            else:
                clip = clips['three_asks', style]
                source = dict(audio_file=clip['audio_file'], audio_sha256=clip['audio_sha256'],
                              spoken_prompt=spoken, seconds=clip['seconds'])
            records.append(dict(question_id=q, condition=style, style=style, voice='en-US-JennyNeural',
                                reference_answer=item['answer'], prompt=prompt if experiment=='phi' else item['question'],
                                **source))
    plan = dict(protocol='multiarith-three-ask-replication-v1', task='multiarith', name='multiarith',
                model=experiment, styles=list(styles), bank=bank, records=records, recipe=RECIPE,
                primary_metric='all-three-observed', max_new_tokens=512, seed=42, written_reference_prompt=prompt,
                instruction_checks_sha256=winners.digest(Path(__file__).with_name('multiarith_instruction_checks.py')))
    winners.freeze(folder/'run_config.json', plan)
    return plan


def generate_phi(folder, plan, native, paths):
    from phi4_all_layer_diagnostic import generate
    done = saved(folder, plan, repair=True)
    checksum = winners.digest(folder/'run_config.json')
    with (folder/'deliveries.jsonl').open('a') as out:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key in done:
                continue
            runtime_record = dict(record, audio_file=str(paths[record['audio_sha256']]))
            trace = generate(native, runtime_record, record['prompt'], interventions=[], max_new_tokens=plan['max_new_tokens'])
            row = dict(record, **trace, run_config_sha256=checksum, **plan['recipe'])
            out.write(json.dumps(row)+'\n')
            out.flush()
            done[key] = row
            print(f'Phi {len(done)}/{len(plan["records"])} {key[0]} {key[1]} {row["response_tokens"]} tokens', flush=True)


def report(folder, plan):
    if winners.digest(Path(__file__).with_name('multiarith_instruction_checks.py')) != plan['instruction_checks_sha256']:
        raise ValueError('Three-ask scoring rules changed.')
    done = saved(folder, plan)
    styles = plan['styles']
    ids = [q for q in plan['bank'] if all((q, s) in done for s in styles)]
    print(f'{plan["model"]}: {len(done)}/{len(plan["records"])} responses, {len(ids)}/290 matched questions.')
    if not ids:
        return
    rows = [done[q,s] for q in ids for s in styles]
    scored, provenance = winners.score_rows({'target': plan}, rows, folder/'scoring')
    groups = {s: {} for s in styles}
    for row in scored:
        row['metrics'].update(checks(row['response']))
        groups[row['condition']][row['question_id']] = row
    lengths, comparisons = {}, []
    for s in styles:
        tokens = [r['response_tokens'] for r in groups[s].values()]
        lengths[s] = dict(mean=mean(tokens), median=median(tokens))
        print(f'{s}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    metrics = list(scored[0]['metrics'])
    metrics.remove(plan['primary_metric'])
    for metric in [plan['primary_metric'], *metrics]:
        for s in styles[1:]:
            stats = paired([(groups['neutral'][q]['metrics'][metric], groups[s][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(emotion=s, metric=metric, **stats))
            print(f'{metric}: neutral {stats["baseline_correct"]}/{len(ids)} -> {s} {stats["steered_correct"]}/{len(ids)}, '
                  f'{stats["gain_pp"]:+.2f} pp, 95% CI {stats["ci95_gain_pp"]}, p={stats["mcnemar_exact_p"]:.4g}')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), comparisons=comparisons,
                        token_lengths=lengths, scorer=provenance), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['phi', 'diffa-flip'])
    parser.add_argument('mode', choices=['prepare','run','summary'])
    args = parser.parse_args()
    folder = Path('outputs')/f'multi3-{args.experiment}290'
    config = folder/'run_config.json'
    if args.mode == 'summary':
        if config.is_file():
            report(folder, json.loads(config.read_text()))
        else:
            print(f'{folder}: awaiting preparation.')
        return
    with job_lock(folder):
        plan = prepare(folder, args.experiment)
        print(plan['records'][0].get('spoken_prompt', plan['written_reference_prompt']), flush=True)
        if args.experiment == 'diffa-flip':
            for record in plan['records'][:2]:
                print(f'{record["style"]} instruction audio: {record["seconds"]:.1f}s')
        done = saved(folder, plan, repair=True)
        pending = [r for r in plan['records'] if (r['question_id'], r['condition']) not in done]
        if pending and args.experiment == 'phi':
            paths = resolve_phi_recordings(pending, folder)
            print(f'Verified original audio for {len(pending)} pending Phi responses.', flush=True)
        if args.mode == 'prepare' or not pending:
            return
        if args.experiment == 'phi':
            from phi4_more_bare_eval import model_session
            with model_session() as native:
                generate_phi(folder, plan, native, paths)
        else:
            from diffa_model_receipts import model_revisions
            from emotion_block_night import DiffaBackend
            from phi4_queued_workers import gpu_queue
            from src.dataloader import build_chat_prompt
            with gpu_queue():
                backend = DiffaBackend()
                winners.freeze(folder/'model-revisions.json', model_revisions())
                generate_flipped(folder, plan, backend, build_chat_prompt)


if __name__ == '__main__':
    main()
