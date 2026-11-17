"""Full BBH-five neutral/angry evaluation on native candidate models; both prompts, no steering."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from bbh_candidate_adapters import MODELS
import evaluate_bbh_candidates as candidates
import evaluate_bbh_audio_models as bbh
from kimi_bbh_layer_diagnostic import digest, freeze
from run_kimi_bbh5 import SOURCE_SHA256, SOURCE_URL, PROMPT, EXPLICIT_PROMPT
from run_kimi_gsm8k import prepare_audio
import run_reasoning_transfer as reasoning
from steer import read_rows


def bank(source):
    path = source/'logical_deduction_five_objects-source.json'
    if digest(path) != SOURCE_SHA256:
        raise ValueError('BBH-five source checksum differs.')
    expected = {q['question_id']: q for q in reasoning.bbh_questions(path.read_bytes(), 'five')}
    questions = read_rows(source/'questions.jsonl')
    if len(questions) != 250 or {q['question_id']: q for q in questions} != expected:
        raise ValueError('Need all 250 unchanged BBH-five questions.')
    return questions


def audio(args):
    if (args.source/'plan.jsonl').exists():
        try:
            make_plan(args.source, args.model, 'original')
        except FileNotFoundError:
            pass
        else:
            print('All 500 existing BBH-five clips verified; no TTS needed.', flush=True); return
    args.source.mkdir(parents=True, exist_ok=True)
    path = args.source/'logical_deduction_five_objects-source.json'
    if not path.exists():
        cached = next((p/path.name for p in args.audio_sources if (p/path.name).exists()), None)
        data = cached.read_bytes() if cached else reasoning.download(SOURCE_URL)
        import hashlib
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('BBH-five download checksum differs.')
        path.write_bytes(data)
    questions = read_rows(args.source/'questions.jsonl')
    if not questions:
        import random
        questions = reasoning.bbh_questions(path.read_bytes(), 'five')
        random.Random(42).shuffle(questions)
        reasoning.bank_file(args.source/'questions.jsonl', questions)
    questions = bank(args.source)
    prepare_audio(args, dict(collection=[], test=questions))


def make_plan(source, model, variant):
    questions = bank(source); by_id = {q['question_id']: q for q in questions}
    records = reasoning.source_records(source, by_id, ['neutral', 'angry'])
    tasks = []
    for q in questions:
        pair = [records[q['question_id'], s] for s in ('neutral', 'angry')]
        if pair[0]['voice'] != pair[1]['voice']:
            raise ValueError('Neutral/angry voices differ.')
        for r in pair:
            task = dict(r, condition=r['style'], reference_answer=r['answer']); task.pop('answer'); tasks.append(task)
    if model == 'flamingo' and any(r['seconds'] > 600 for r in tasks):
        raise ValueError('Refusing to truncate audio beyond the native AF3 limit.')
    return dict(task='bbh_cot_zeroshot_logical_deduction_five_objects', protocol='bbh-five-candidate-delivery-v1',
        model_key=model, source=str(source.absolute()), source_sha256=SOURCE_SHA256, bank=by_id, recordings=tasks,
        prompt=PROMPT if variant == 'original' else EXPLICIT_PROMPT, prompt_variant=variant,
        recipe=dict(MODELS[model], dtype='bfloat16', do_sample=False, num_beams=1, use_cache=True,
                    repetition_penalty=1., text_only_output=True), seed=42, max_new_tokens=512,
        selection='All 250 BBH-five questions, both deliveries; no collection, steering or outcome/classifier/duration filter.',
        interventions=[], audio_encoding='Full mono 16kHz audio through native candidate processor/encoder.',
        duration_audit=dict(max_seconds=max(r['seconds'] for r in tasks), over_40_seconds=sum(r['seconds'] > 40 for r in tasks)))


def prepare(args):
    plans = []
    for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
        output = args.output/args.model/variant; output.mkdir(parents=True, exist_ok=True)
        plan = make_plan(args.source, args.model, variant); freeze(output/'run_config.json', plan)
        plans.append((output, plan))
        print(f'{args.model}/{variant}: all 250 BBH-five questions, neutral/angry; 500 answers, no steering.', flush=True)
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=tuple(MODELS), default='phi4')
    p.add_argument('--mode', choices=('audio', 'prepare', 'evaluate', 'summary'), default='evaluate')
    p.add_argument('--prompt', choices=('original', 'explicit', 'both'), default='both')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-bbh-five250'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-five-candidates'))
    p.add_argument('--audio-sources', nargs='+', type=Path, default=[Path('outputs/azure-bbh-five250')])
    p.add_argument('--max-generations', type=int, default=0)
    args = p.parse_args()
    if args.max_generations < 0: p.error('Generation limit must be nonnegative.')
    if args.mode == 'audio': audio(args); return
    if args.mode == 'summary':
        for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
            bbh.report(args.output/args.model/variant, objects='five')
        return
    plans = prepare(args)
    if args.mode == 'prepare': return
    with (args.output/args.model/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            candidates.run(plans, args.model, args.max_generations)


if __name__ == '__main__': main()
