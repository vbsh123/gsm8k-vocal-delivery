"""Queue BBH-five transfer, SVAMP mean collection/held-out steering, and aligned-prompt GSM8K."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import wave

from compare_cloud_accuracy import PROMPT as GSM_COLLECTION_PROMPT
from extract_audio_vectors import GROUPS, save_tensor
from rescore_audio_steering import HarnessTaskScorer, download, numeric_equal, numeric_within_tolerance, paired
from run_bbh_logical200 import REVISION as BBH_REVISION, PROMPT as BBH_COLLECTION_PROMPT
from run_svamp500 import HASH_PROMPT, SOURCE_SHA256 as SVAMP_SHA, select_questions as svamp_questions
from score_bbh_harness import HarnessScorer
from steer import install_interventions, read_rows

STAGES = ('bbh5', 'svamp', 'gsm8k')
STEERED = 'neutral-L31-answer-a1'
BBH_PROMPT = BBH_COLLECTION_PROMPT.replace('three choices A, B, C', 'five choices A, B, C, D, E').replace(
    'correct option A, B, or C', 'correct option A, B, C, D, or E')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    text = json.dumps(value, indent=2)+'\n'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Experiment settings changed: {path}. Use another output folder.')
    path.write_text(text)


def bank_file(path, questions):
    text = ''.join(json.dumps(q)+'\n' for q in questions)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Question selection changed: {path}')
    path.write_text(text)


def generate_audio(source, questions, styles):
    source.mkdir(parents=True, exist_ok=True)
    bank_file(source/'questions.jsonl', questions)
    subprocess.run([sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5',
        '--input-jsonl', str(source/'questions.jsonl'), '--num-examples', str(len(questions)),
        '--styles', *styles, '--output', str(source), '--resume', '--retry-passes', '5'], check=True)


def recording(row, source):
    path = (source/row['audio_path']).absolute()
    with wave.open(str(path), 'rb') as audio:
        seconds = audio.getnframes()/audio.getframerate()
    if seconds <= 0:
        raise ValueError(f'Empty audio: {path}')
    return dict(question_id=row['question_id'], voice=row['voice'], style=row['style'],
        answer=str(row['answer']).rsplit('####', 1)[-1].strip(),
        audio_file=str(path), audio_sha256=digest(path), seconds=seconds)


def source_records(source, bank, styles):
    records = {}
    for row in read_rows(source/'plan.jsonl'):
        if row['provider'] != 'azure' or row['style'] not in styles:
            continue
        q = bank[row['question_id']]
        if row['question'] != q['question'] or str(row['answer']) != str(q['answer']):
            raise ValueError('Audio plan differs from the question bank.')
        key = row['question_id'], row['style']
        if key in records:
            raise ValueError('Duplicate question/delivery.')
        records[key] = recording(row, source)
    if set(records) != {(qid, s) for qid in bank for s in styles}:
        raise ValueError('Missing requested recordings.')
    return records


def tasks_for(records, ids, angry=True):
    return [dict(records[qid, 'angry' if c == 'angry' else 'neutral'], condition=c)
            for qid in ids for c in (('neutral', 'angry', STEERED) if angry else ('neutral', STEERED))]


def bbh_questions(data):
    examples = json.loads(data)['examples']
    if len(examples) != 250:
        raise ValueError('Expected 250 official BBH five-object questions.')
    questions = []
    for i, row in enumerate(examples):
        stem, options = row['input'].rsplit('\nOptions:\n', 1)
        choices = [dict(label=m[1], text=m[2]) for line in options.splitlines()
                   if (m := re.fullmatch(r'\(([A-E])\) (.+)', line))]
        if [c['label'] for c in choices] != list('ABCDE') or row['target'] not in [f'({c})' for c in 'ABCDE']:
            raise ValueError('Unexpected BBH five-object options.')
        questions.append(dict(question_id=f'bbh_logical_five_{i:03d}', source_index=i,
            original_input=row['input'], original_target=row['target'], answer=row['target'][1], choices=choices,
            question=stem+'\nOptions:\n'+'\n'.join(f'Option {c["label"]}: {c["text"]}' for c in choices)))
    return questions


def prepare_bbh(args, folder):
    vector = args.bbh_vector
    vector_sha256 = digest(vector)
    source = args.bbh_source; source.mkdir(parents=True, exist_ok=True)
    path = source/'logical_deduction_five_objects-source.json'
    if not path.exists():
        path.write_bytes(download(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REVISION}/bbh/logical_deduction_five_objects.json'))
    questions = bbh_questions(path.read_bytes())
    bank = {q['question_id']: q for q in questions}
    generate_audio(source, questions, ['neutral', 'angry'])
    records = source_records(source, bank, ['neutral', 'angry'])
    return dict(task='bbh5', prompt=BBH_PROMPT, collection_prompt=BBH_COLLECTION_PROMPT,
        vector=str(vector.absolute()), vector_sha256=vector_sha256, bank=bank,
        source_sha256=digest(path), extraction=[], recordings=tasks_for(records, bank))


def svamp_split(data, existing):
    if hashlib.sha256(data).hexdigest() != SVAMP_SHA:
        raise ValueError('SVAMP source checksum differs.')
    all_questions = svamp_questions(data, 1000)
    expected = {q['question_id']: q for q in all_questions[:500]}
    if len(existing) != 500 or {q['question_id']: q for q in existing} != expected:
        raise ValueError('Expected the existing first 500 SVAMP questions; extraction selection is unchanged.')
    return all_questions[:500], all_questions[500:]


def prepare_svamp(args, folder):
    data = (args.svamp_existing/'SVAMP-source.json').read_bytes()
    train, test = svamp_split(data, read_rows(args.svamp_existing/'questions.jsonl'))
    train_bank = {q['question_id']: q for q in train}
    records = source_records(args.svamp_existing, train_bank, ['neutral', 'angry'])
    # Collection uses exactly the existing native SVAMP audio; no length filtering changes the 500 pairs.
    if any(r['seconds'] > 30 for r in records.values()):
        raise ValueError('Existing extraction audio exceeds the previous native SVAMP protocol; inspect before collecting.')
    pairs = [dict(question_id=q, neutral=records[q, 'neutral'], angry=records[q, 'angry']) for q in train_bank]
    generate_audio(args.svamp_new_source, test, ['neutral'])
    bank = {q['question_id']: q for q in test}
    test_records = source_records(args.svamp_new_source, bank, ['neutral'])
    return dict(task='svamp', prompt=HASH_PROMPT, collection_prompt=HASH_PROMPT,
        vector=str((folder/'style-vectors.pt').absolute()), bank=bank, extraction=pairs,
        source_sha256=SVAMP_SHA, recordings=tasks_for(test_records, bank, angry=False))


def gsm_sources(args):
    configs = [json.loads((p/'run_config.json').read_text()) for p in args.gsm_cohorts]
    records = [r for config in configs for r in config['recordings']]
    if len(records) != 718 or len({r['question_id'] for r in records}) != 718:
        raise ValueError('Need the same 718 distinct neutral recordings from the completed GSM8K steering runs.')
    vector = args.gsm_vector or Path(configs[0]['settings']['vectors'])
    if args.gsm_vector is None and any(Path(c['settings']['vectors']).absolute() != vector.absolute() for c in configs):
        raise ValueError('GSM8K cohort configs reference different vectors.')
    return records, vector


def prepare_gsm(args, folder):
    selected, vector = gsm_sources(args)
    official_path = folder/'gsm8k-test.jsonl'
    if not official_path.exists():
        official_path.write_bytes(download('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'))
    official = read_rows(official_path)
    wanted = {}
    for r in selected:
        voice = r.get('speaker', r.get('voice'))
        wanted[r['question_id'], voice] = r
    found = {}; bank = {}
    for source in args.gsm_audio_sources:
        for row in read_rows(source/'plan.jsonl'):
            key = row['question_id'], row['voice']
            if row['provider'] != 'azure' or row['style'] not in ('neutral', 'angry') or key not in wanted:
                continue
            index = int(re.fullmatch(r'gsm8k_test_(\d+)', row['question_id'])[1])
            q = official[index]; gold = q['answer'].rsplit('####', 1)[-1].strip()
            if row['question'] != q['question'] or str(row['answer']).rsplit('####', 1)[-1].strip() != gold:
                raise ValueError('Azure GSM8K plan differs from official question/reference.')
            record = recording(row, source)
            original = wanted[key]
            if str(original['answer']).rsplit('####', 1)[-1].strip() != gold:
                raise ValueError('Original steering gold differs from GSM8K source.')
            if row['style'] == 'neutral' and digest(original['audio_path']) != record['audio_sha256']:
                raise ValueError('Neutral audio is not the same original steering recording.')
            record_key = row['question_id'], row['style']
            if record_key in found:
                raise ValueError('Duplicate GSM8K question/style in audio sources.')
            found[record_key] = record
            bank[row['question_id']] = dict(question_id=row['question_id'], question=q['question'], answer=gold)
    ids = [r['question_id'] for r in selected]
    if set(found) != {(q, s) for q in ids for s in ('neutral', 'angry')}:
        missing = [(q, s) for q in ids for s in ('neutral', 'angry') if (q, s) not in found]
        raise ValueError(f'Missing original GSM8K audio: {missing[:5]}; pass --gsm-audio-sources with both cohort folders.')
    return dict(task='gsm8k', prompt=HASH_PROMPT, collection_prompt=GSM_COLLECTION_PROMPT,
        vector=str(vector.absolute()), vector_sha256=digest(vector), bank=bank, extraction=[],
        source_sha256=digest(official_path), recordings=tasks_for(found, ids))


class NumericScorer:
    """Official GSM8K task scoring; numeric equivalence is an additional metric."""
    def __init__(self, cache):
        self.harness = HarnessTaskScorer('gsm8k', cache)
        self.provenance = dict(self.harness.provenance,
            numeric_equivalence='Additional custom Fraction comparison; not an official SVAMP metric')

    def score(self, response, gold):
        values = self.harness.score([response], [dict(question='Saved spoken math question', answer=gold)])
        predictions = {name: rows[0]['prediction'] for name, rows in values.items()}
        metrics = {name: rows[0]['correct'] for name, rows in values.items()}
        metrics['flexible-numeric'] = numeric_equal(predictions['flexible-extract'], gold)
        metrics['format-compliant'] = predictions['strict-match'] != '[invalid]'
        return metrics, predictions


def scorer_for(plan, folder):
    cache = folder/'harness/upstream'
    if plan['task'] == 'bbh5':
        return HarnessScorer(cache, objects='five')
    return NumericScorer(cache)


def score_response(scorer, plan, row):
    q = plan['bank'][row['question_id']]
    if row['reference_answer'] != q['answer']:
        raise ValueError('Saved answer reference differs from frozen source.')
    if plan['task'] == 'bbh5':
        values = scorer.score([row['response']], [dict(input=q['original_input'], target=q['original_target'])])
        return {name: v[0]['correct'] for name, v in values.items()}, {name: v[0]['prediction'] for name, v in values.items()}
    metrics, predictions = scorer.score(row['response'], q['answer'])
    if plan['task'] == 'svamp':
        metrics['author-tolerance'] = numeric_within_tolerance(predictions['flexible-extract'], q['answer'])
    return metrics, predictions


def report(folder):
    path = folder/'run_config.json'
    if not path.exists():
        print(f'{folder.name}: not prepared.', flush=True); return
    plan = json.loads(path.read_text()); scorer = scorer_for(plan, folder)
    rows = read_rows(folder/'results.jsonl'); groups = {}; scored = []
    expected = {(r['question_id'], r['voice'], r['condition']) for r in plan['recordings']}
    for row in rows:
        if (row['question_id'], row['voice'], row['condition']) not in expected:
            raise ValueError('Unexpected evaluation question/condition.')
        metrics, predictions = score_response(scorer, plan, row)
        groups.setdefault(row['condition'], {})[row['question_id'], row['voice']] = metrics
        scored.append(dict(row, scores=metrics, predictions=predictions))
    print(f'\n{plan["task"]}: {len(rows)}/{len(expected)} evaluation answers saved. L31 answer x1; seed42.', flush=True)
    summary = dict(task=plan['task'], scorer=scorer.provenance, primary_metric='flexible-extract' if plan['task'] == 'bbh5' else 'flexible-numeric',
        accuracy=[], paired=[], planned_answers=len(expected), saved_answers=len(rows))
    metrics = next(iter(next(iter(groups.values())).values())) if groups else {}
    for metric in metrics:
        print(f'\n{metric}')
        for condition, answers in groups.items():
            correct = sum(v[metric] for v in answers.values()); n = len(answers)
            print(f'  {condition}: {correct}/{n} ({correct/n:.1%})')
            summary['accuracy'].append(dict(metric=metric, condition=condition, n=n, correct=correct))
        for condition, answers in groups.items():
            if condition == 'neutral':
                continue
            base = groups.get('neutral', {}); keys = sorted(base.keys() & answers.keys())
            if keys:
                result = paired([(base[k][metric], answers[k][metric]) for k in keys])
                summary['paired'].append(dict(metric=metric, condition=condition, **result))
                lo, hi = result['ci95_gain_pp']
                print(f'  {condition} vs neutral: N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; '
                    f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}]')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (folder/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    print('Audio-adapted evaluation, not full official text-harness inference. P-values/intervals unadjusted.', flush=True)
    if plan['task'] == 'gsm8k':
        print('The fixed GSM8K vector was collected under the previous prompt; this tests transfer to the aligned #### prompt.', flush=True)


def evaluate(plan, folder):
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen_chunked, load_model

    completed = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(folder/'results.jsonl', repair=True)}
    needed = [r for r in plan['recordings'] if (r['question_id'], r['voice'], r['condition']) not in completed]
    pairs_dir = folder/'pairs'
    if plan['extraction']:
        pairs_dir.mkdir(exist_ok=True)
    pending = [(i, p) for i, p in enumerate(plan['extraction']) if not (pairs_dir/f'{i:05d}.pt').exists()]
    if not pending and not needed and Path(plan['vector']).exists():
        return
    scorer = scorer_for(plan, folder)
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    # Load frozen vector and check disjointness before loading the model.
    state = None
    if not plan['extraction']:
        if digest(plan['vector']) != plan['vector_sha256']:
            raise ValueError('Frozen vector changed.')
        state = torch.load(plan['vector'], map_location='cpu', weights_only=True)
        if state['prompt'] != plan['collection_prompt'] or state['max_new_tokens'] != 512 or state.get('mode', 'fixed') != 'fixed':
            raise ValueError('Expected the original fixed mean vector and original collection prompt.')
        if set(state.get('extraction_question_ids', [])) & set(plan['bank']):
            raise ValueError('Vector extraction and evaluation questions overlap.')
    model, tokenizer, features = load_model(); layers = model.llm_model.model.transformer.blocks

    def generate(task):
        if digest(task['audio_file']) != task['audio_sha256']:
            raise ValueError('Audio changed since preparation.')
        audio, sr = sf.read(task['audio_file'], dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        return gen_chunked(audio, sr, model, tokenizer, features, plan['prompt'], 512, 30)

    def capture(task):
        states = []
        def hook(module, inputs, result):
            states.append(result[0][0, -512:].float().mean(0).cpu())
        handle = layers[31].register_forward_hook(hook)
        try:
            response, _ = generate(task)
        finally:
            handle.remove()
        return response, torch.stack(states)

    for i, pair in pending:
        print(f'Collect [{i+1}/{len(plan["extraction"])}] {pair["question_id"]} neutral + angry', flush=True)
        nr, n = capture(pair['neutral']); ar, a = capture(pair['angry']); steps = min(len(n), len(a))
        save_tensor(torch, dict(question_id=pair['question_id'], difference=(a[:steps]-n[:steps]).mean(0),
            neutral_response=nr, angry_response=ar, neutral_forwards=len(n), angry_forwards=len(a), matched_forwards=steps), pairs_dir/f'{i:05d}.pt')
    if plan['extraction']:
        values = [torch.load(pairs_dir/f'{i:05d}.pt', map_location='cpu', weights_only=True) for i in range(len(plan['extraction']))]
        if any(v['question_id'] != p['question_id'] for v, p in zip(values, plan['extraction'])):
            raise ValueError('Extraction pair cache differs from frozen split.')
        total = torch.stack([v['difference'] for v in values]).sum(0)
        sums = torch.zeros(len(layers), len(GROUPS), len(total)); sums[31, GROUPS.index('answer')] = total
        counts = torch.zeros(len(layers), dtype=torch.long); counts[31] = len(values)
        state = dict(pair_mean_sums={'angry': sums}, pair_counts={'angry': counts}, groups=GROUPS,
            prompt=plan['collection_prompt'], seed=42, max_new_tokens=512, mode='fixed', method='mean',
            collected_targets=[(31, 'answer')], extraction_question_ids=[p['question_id'] for p in plan['extraction']],
            averaging='Mean matched-forward angry-minus-neutral difference per question, then equal question weight')
        save_tensor(torch, state, Path(plan['vector']))
    count = int(state['pair_counts']['angry'][31])
    if count <= 0:
        raise ValueError('Missing L31 mean direction.')
    direction = state['pair_mean_sums']['angry'][31, state['groups'].index('answer')]/count
    directions = {(31, 'answer'): direction.to(device='cuda', dtype=torch.bfloat16)}
    del state
    for i, task in enumerate(needed, 1):
        interventions = [dict(layer=31, group='answer', strength=1)] if task['condition'] == STEERED else []
        print(f'Evaluate [{i}/{len(needed)}] {task["question_id"]} {task["condition"]}', flush=True)
        started = time.monotonic()
        with install_interventions(layers, directions, interventions, {}, 512) as states:
            response, encoding = generate(task)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'],
            condition=task['condition'], audio_file=task['audio_file'], audio_sha256=task['audio_sha256'],
            audio_encoding=encoding, reference_answer=task['answer'], response=response, prompt=plan['prompt'],
            seed=42, max_new_tokens=512, steps=128, block_length=32, decoder='fast_dllm',
            interventions=interventions, steered_forwards=max(s['state']['steered'] for s in states),
            generation_seconds=time.monotonic()-started)
        row['scores'], row['predictions'] = score_response(scorer, plan, row)
        with (folder/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f'  {row["scores"]}\n{response}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', *STAGES, 'summary'], default='all')
    p.add_argument('--output', type=Path, default=Path('outputs/reasoning-transfer'))
    p.add_argument('--bbh-source', type=Path, default=Path('outputs/azure-bbh-five250'))
    p.add_argument('--bbh-vector', type=Path, default=Path('outputs/bbh-logical250-angry-mean/style-vectors.pt'))
    p.add_argument('--svamp-existing', type=Path, default=Path('outputs/azure-svamp500'))
    p.add_argument('--svamp-new-source', type=Path, default=Path('outputs/azure-svamp-remaining500'))
    p.add_argument('--gsm-vector', type=Path)
    p.add_argument('--gsm-cohorts', nargs='+', type=Path, default=[Path('outputs/steer-jenny-late-layers/original_mean'),
        Path('outputs/steer-jenny-late-layers/third-cohort/original_mean')])
    p.add_argument('--gsm-audio-sources', nargs='+', type=Path, default=[Path('outputs/azure-jenny-next500'),
        Path('outputs/azure-jenny-remaining-test')])
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'summary':
        for stage in STAGES:
            report(args.output/stage)
        return
    if args.stage == 'all':
        # Fail before TTS spending if a required existing cohort or vector is absent.
        digest(args.bbh_vector)
        svamp_split((args.svamp_existing/'SVAMP-source.json').read_bytes(),
                    read_rows(args.svamp_existing/'questions.jsonl'))
        _, gsm_vector = gsm_sources(args)
        digest(gsm_vector)
        for source in args.gsm_audio_sources:
            if not (source/'plan.jsonl').exists():
                raise FileNotFoundError(source/'plan.jsonl')
        # Each subprocess exits before the next model loads, releasing all GPU memory.
        with (args.output/'queue.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for stage in STAGES:
                print(f'\nSTAGE: {stage}', flush=True)
                freeze_path = args.output/'status.json'
                freeze_path.write_text(json.dumps(dict(stage=stage, status='running'))+'\n')
                command = [sys.executable, '-u', str(Path(__file__).absolute()), *sys.argv[1:], '--stage', stage]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError:
                    freeze_path.write_text(json.dumps(dict(stage=stage, status='failed'))+'\n')
                    raise
            (args.output/'status.json').write_text(json.dumps(dict(stage='finished', status='complete'))+'\n')
        return
    folder = args.output/args.stage; folder.mkdir(parents=True, exist_ok=True)
    with (folder/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare = {'bbh5': prepare_bbh, 'svamp': prepare_svamp, 'gsm8k': prepare_gsm}[args.stage]
        plan = prepare(args, folder)
        plan.update(seed=42, max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm',
            layer=31, group='answer', strength=1, audio_encoding='Native up to 30s; full unchanged longer audio uses 30s chunked Whisper and global adapters')
        train_ids = {r['question_id'] for r in plan['extraction']}
        if train_ids & set(plan['bank']):
            raise ValueError('Extraction/test questions overlap.')
        freeze(folder/'run_config.json', plan)
        scorer_for(plan, folder)  # Verify scorer dependencies before loading the GPU model.
        long = {(r['question_id'], r['style']) for r in plan['recordings'] if r['seconds'] > 30}
        print(f'{args.stage}: {len(plan["bank"])} test questions; {len(plan["extraction"])} collection pairs; '
              f'{len(plan["recordings"])} planned answers; {len(long)} long recordings require full-audio chunking.', flush=True)
        for kind in ('svamp', 'bbh', 'finalist-queue', 'azure-emotions'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            evaluate(plan, folder)
        report(folder)


if __name__ == '__main__':
    main()
