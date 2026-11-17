"""BBH three-object neutral/angry delivery evaluation with MOSS-Audio or Kimi-Audio. No steering."""

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.request import urlopen
import wave

from run_bbh_logical200 import PROMPT, SOURCE_SHA256, SOURCE_URL, select_questions
from steer import read_rows

MOSS_MODEL = 'OpenMOSS-Team/MOSS-Audio-8B-Instruct'
MOSS_REVISION = 'd4dc5a6d8cd79b43dcd82884c75e303b1ecd016d'
MOSS_SOURCE_REVISION = '66326e6e0db34f036c86a76ba005efa4830c69dd'
STYLES = ('neutral', 'angry')
FORMAT_METRIC = 'strict-format-extractable'
EXPLICIT_REASONING_PROMPT = PROMPT.replace("Let's think step by step.",
    'Give concise step-by-step reasoning with only the necessary deductions.')


def generate_audio(source):
    from run_reasoning_transfer import generate_audio as azure_generate
    source.mkdir(parents=True, exist_ok=True)
    original = source/'logical_deduction_three_objects-source.json'
    if not original.exists():
        with urlopen(SOURCE_URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('Official BBH source checksum differs.')
        original.write_bytes(data)
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official BBH source checksum differs.')
    print('Generate 250 BBH questions × neutral/angry = 500 Azure clips; Jenny, intensity 1.5.', flush=True)
    azure_generate(source, select_questions(data, 250), list(STYLES))


def source_folder(explicit):
    if explicit:
        return explicit
    for path in (Path('outputs/azure-bbh-logical250-steering'), Path('outputs/azure-bbh-logical200')):
        if (path/'questions.jsonl').exists():
            return path
    raise FileNotFoundError('BBH three-object audio not found. Pass --source /path/to/azure-bbh-logical250-steering.')


def make_plan(source, model):
    original = (source/'logical_deduction_three_objects-source.json').read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official BBH source checksum differs.')
    questions = read_rows(source/'questions.jsonl')
    if questions != select_questions(original, len(questions)):
        raise ValueError('Existing question selection differs from the seeded BBH cohort.')
    bank = {q['question_id']: q for q in questions}
    groups = {}
    for row in read_rows(source/'plan.jsonl'):
        if row['provider'] != 'azure' or row['style'] not in STYLES:
            continue
        q = bank[row['question_id']]
        if row['question'] != q['question'] or row['answer'] != q['answer']:
            raise ValueError('Audio wording/reference differs from official BBH adaptation.')
        path = (source/row['audio_path']).absolute()
        with wave.open(str(path), 'rb') as audio:
            seconds = audio.getnframes()/audio.getframerate()
        if seconds <= 0:
            raise ValueError('Empty BBH recording.')
        key = row['question_id'], row['voice']
        group = groups.setdefault(key, {})
        if row['style'] in group:
            raise ValueError('Duplicate BBH recording.')
        group[row['style']] = dict(question_id=key[0], voice=key[1], style=row['style'], condition=row['style'],
            audio_file=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            seconds=seconds, reference_answer=q['answer'])
    if {q for q, voice in groups} != set(bank) or any(set(g) != set(STYLES) for g in groups.values()):
        raise ValueError('Complete neutral/angry audio pairs are required for every question.')
    recipe = dict(model=MOSS_MODEL, revision=MOSS_REVISION, source_revision=MOSS_SOURCE_REVISION,
                  transformers='4.57.1', attention='sdpa', enable_time_marker=True,
                  dtype='bfloat16', do_sample=False, num_beams=1, use_cache=True)
    if model == 'kimi':
        # Imported only in the Kimi environment; model loading happens later under the GPU lock.
        from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
        recipe = dict(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION,
            tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
            transformers='4.51.3', output_type='text', load_detokenizer=False,
            text_temperature=0., audio_temperature=0., text_top_k=5, audio_top_k=5,
            text_repetition_penalty=1., audio_repetition_penalty=1.)
    return dict(task='bbh_cot_zeroshot_logical_deduction_three_objects', source=str(source.absolute()),
        source_sha256=SOURCE_SHA256, bank=bank, recordings=[g[s] for g in groups.values() for s in STYLES],
        prompt=PROMPT, seed=42, max_new_tokens=512, styles=list(STYLES), recipe=recipe,
        selection='All existing seeded questions; matched voice; no outcome/classifier/30-second filtering',
        audio_encoding='Full recording through the model authors\' own processor/encoder; no custom DIFFA chunking',
        interventions=[])


def moss_generator(source, prompt=PROMPT):
    import torch
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != MOSS_SOURCE_REVISION:
        raise ValueError('MOSS source differs from the pinned version; run setup_moss_audio.sh.')
    sys.path.insert(0, str(source.absolute()))
    from moss_audio_io import load_audio
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor
    model = MossAudioModel.from_pretrained(MOSS_MODEL, revision=MOSS_REVISION,
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', trust_remote_code=True).eval()
    processor = MossAudioProcessor.from_pretrained(MOSS_MODEL, revision=MOSS_REVISION,
        trust_remote_code=True, enable_time_marker=True)

    def generate(task):
        audio = load_audio(task['audio_file'], sample_rate=processor.config.mel_sr)
        inputs = processor(text=prompt, audios=[audio], return_tensors='pt').to(model.device)
        if inputs.get('audio_data') is not None:
            inputs['audio_data'] = inputs['audio_data'].to(model.dtype)
        inputs['audio_input_mask'] = inputs['input_ids'] == processor.audio_token_id
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True)
        suffix = ids[0, inputs['input_ids'].shape[1]:]
        return processor.decode(suffix, skip_special_tokens=True), len(suffix)
    return generate


def kimi_generator(source, prompt=PROMPT):
    import torch
    from evaluate_kimi_audio import assets, messages
    model_path, tokenizer_path, KimiAudio = assets(source)
    from kimia_infer.api.prompt_manager import Glm4Tokenizer
    with patch('kimia_infer.api.prompt_manager.Glm4Tokenizer', side_effect=lambda _: Glm4Tokenizer(tokenizer_path)):
        model = KimiAudio(model_path=model_path, load_detokenizer=False)
    model.alm.eval()

    def generate(task):
        with torch.inference_mode():
            _, response = model.generate(messages(Path(task['audio_file']), prompt), output_type='text',
                max_new_tokens=512, text_temperature=0., audio_temperature=0., text_top_k=5, audio_top_k=5,
                text_repetition_penalty=1., audio_repetition_penalty=1.)
        return response, None
    return generate


def run(plan, output, model, source, max_generations=0):
    done = {(r['question_id'], r['voice'], r['style']) for r in read_rows(output/'results.jsonl', repair=True)}
    pending = [r for r in plan['recordings'] if (r['question_id'], r['voice'], r['style']) not in done]
    if max_generations:
        pending = pending[:max_generations]
    print(f'{model}: {len(plan["recordings"])} planned answers; {len(pending)} pending in this invocation; NO steering.', flush=True)
    if not pending:
        return
    import numpy as np
    import torch
    (output/'runtime.json').write_text(json.dumps(dict(python=sys.version, torch=torch.__version__,
        numpy=np.__version__, transformers=version('transformers'),
        audio_loader='soundfile-float32-moss-mono-torchaudio-resample-v1' if model == 'moss' else 'upstream-kimi'), indent=2)+'\n')
    generate = moss_generator(source, plan['prompt']) if model == 'moss' else kimi_generator(source, plan['prompt'])
    for i, task in enumerate(pending, 1):
        if hashlib.sha256(Path(task['audio_file']).read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('Audio changed after preparation.')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        print(f'[{i}/{len(pending)}] {task["question_id"]} {task["style"]} ({task["seconds"]:.2f}s)', flush=True)
        started = time.monotonic()
        response, tokens = generate(task)
        row = dict(task, response=response, response_tokens=tokens, prompt=plan['prompt'], seed=42,
                   model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                   max_new_tokens=512, generation_seconds=time.monotonic()-started, interventions=[])
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(response, flush=True)


def strict_format_bits(scorer, responses, docs, letters='ABC'):
    """Would upstream strict scoring accept any valid choice? Ignore the true answer."""
    # Reuse both upstream extraction AND metric normalization. This is an added
    # diagnostic; the real-reference accuracy calculation remains unchanged.
    expanded = [response for response in responses for _ in letters]
    references = [dict(doc, target=f'({choice})') for doc in docs for choice in letters]
    results = scorer.score(expanded, references)['strict-match']
    count = len(letters)
    return [any(r['correct'] for r in results[i:i+count]) for i in range(0, len(results), count)]


def report(output, objects='three'):
    from score_bbh_harness import HarnessScorer
    from rescore_audio_steering import paired
    if not (output/'run_config.json').exists():
        print(f'{output}: not prepared.', flush=True); return
    plan = json.loads((output/'run_config.json').read_text())
    rows = read_rows(output/'results.jsonl')
    scorer = HarnessScorer(output/'harness/upstream', objects=objects)
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    if any(r['reference_answer'] != plan['bank'][r['question_id']]['answer'] for r in rows):
        raise ValueError('Saved reference differs from the frozen question bank.')
    values = scorer.score([r['response'] for r in rows], docs)
    letters = {'three': 'ABC', 'five': 'ABCDE', 'seven': 'ABCDEFG'}[objects]
    format_bits = strict_format_bits(scorer, [r['response'] for r in rows], docs, letters)
    scored = [dict(row, harness={m: results[i] for m, results in values.items()},
                   **{FORMAT_METRIC: format_bits[i]}) for i, row in enumerate(rows)]
    summary = dict(model=plan['recipe'], scorer=scorer.provenance, primary_metric='flexible-extract', accuracy=[], paired=[],
        format_diagnostic=f'Additional statistic: upstream strict extraction/metric accepts a choice in {letters}, regardless of gold. Not a separate official benchmark metric or a check of exact final-line placement.')
    subsets = {'all': set(plan['bank'])}
    if objects == 'three' and len(plan['bank']) == 250:
        subsets['original-125-test-half'] = set(list(plan['bank'])[125:])
    print(f'\n{plan["recipe"]["model"]}: {len(rows)}/{len(plan["recordings"])} answers; no steering.', flush=True)
    for subset, ids in subsets.items():
        print('\nCohort:', subset, flush=True)
        for metric in [*values, FORMAT_METRIC]:
            groups = {s: {(r['question_id'], r['voice']): (r[FORMAT_METRIC] if metric == FORMAT_METRIC else r['harness'][metric]['correct'])
                          for r in scored if r['style'] == s and r['question_id'] in ids} for s in STYLES}
            keys = sorted(groups['neutral'].keys() & groups['angry'].keys())
            if not keys:
                continue
            for style in STYLES:
                correct = sum(groups[style][k] for k in keys)
                print(f'{metric} / {style}: {correct}/{len(keys)} ({correct/len(keys):.1%})')
                summary['accuracy'].append(dict(cohort=subset, metric=metric, style=style, n=len(keys), correct=correct))
            result = paired([(groups['neutral'][k], groups['angry'][k]) for k in keys])
            summary['paired'].append(dict(cohort=subset, metric=metric, **result))
            lo, hi = result['ci95_gain_pp']
            print(f'  angry vs neutral: {result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                  f'p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}]')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    print('Audio-adapted generations scored verbatim by the installed upstream task API. Tests/intervals unadjusted.')
    print('strict-format-extractable is an additional diagnostic: a valid strict-extracted choice, independent of correctness.')
    if len(subsets) > 1:
        print('The 125-question table is a subset of the full table, not an independent cohort.')
    if rows and all(r.get('response_tokens') is not None for r in rows):
        counts = {s: sum(r['response_tokens'] >= 512 for r in rows if r['style'] == s) for s in STYLES}
        print('Responses reaching the 512-token budget:', counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['moss', 'kimi'], required=True)
    p.add_argument('--mode', choices=['generate', 'prepare', 'evaluate', 'summary'], default='evaluate')
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-three-audio-models'))
    p.add_argument('--model-source', type=Path)
    p.add_argument('--max-generations', type=int, default=0, help='Optional smoke limit; 0 resumes all missing answers')
    p.add_argument('--explicit-reasoning', action='store_true',
                   help='Replace only Let\'s think step by step with an explicit concise reasoning instruction. Use a separate output directory.')
    args = p.parse_args()
    if args.max_generations < 0:
        p.error('--max-generations cannot be negative')
    if args.mode == 'generate':
        generate_audio(args.source or Path('outputs/azure-bbh-three-models-regenerated'))
        return
    output = args.output/args.model; output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(output); return
    plan = make_plan(source_folder(args.source), args.model)
    if args.explicit_reasoning:
        plan['prompt'] = EXPLICIT_REASONING_PROMPT
    config = output/'run_config.json'
    if config.exists() and json.loads(config.read_text()) != plan:
        raise ValueError('Experiment settings changed; use a new output folder.')
    config.write_text(json.dumps(plan, indent=2)+'\n')
    if args.mode == 'prepare':
        print(f'{args.model}: {len(plan["bank"])} matched questions; full recordings, no steering.', flush=True); return
    model_source = args.model_source or Path('external/MOSS-Audio' if args.model == 'moss' else 'external/Kimi-Audio')
    with (output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(plan, output, args.model, model_source, args.max_generations)


if __name__ == '__main__':
    main()
