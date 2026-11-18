"""Matched GSM400 happy versus neutral, native Phi/MiniCPM, no steering."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import evaluate_gsm_candidates as gsm
import evaluate_bbh_candidates as candidates
import run_reasoning_transfer as reasoning
from run_kimi_gsm8k import prepare_audio
from kimi_bbh_layer_diagnostic import digest, freeze
from steer import read_rows

PROTOCOL = 'gsm400-native-neutral-happy-v1'


def selection(): return gsm.questions()[:400]


def audio(args):
    questions = selection()
    args.source.mkdir(parents=True, exist_ok=True)
    freeze(args.source/'selection.json', dict(protocol=PROTOCOL, seed=42, questions=questions,
        question_file_sha256=digest(gsm.QUESTION_FILE), styles=['neutral', 'happy']))
    reasoning.bank_file(args.source/'questions.jsonl', questions)
    prepare_audio(args, dict(collection=[], test=questions), styles=('neutral', 'happy'))


def prepare(args):
    questions = selection(); bank = {q['question_id']: q for q in questions}
    if read_rows(args.source/'questions.jsonl') != questions: raise ValueError('Happy400 question bank differs.')
    records = reasoning.source_records(args.source, bank, ['neutral', 'happy'])
    tasks = []
    for q in questions:
        neutral, happy = (records[q['question_id'], s] for s in ('neutral', 'happy'))
        if neutral['voice'] != happy['voice']: raise ValueError('Voices differ.')
        for row in (neutral, happy):
            task = dict(row, condition=row['style'], reference_answer=row['answer'])
            task.pop('answer'); tasks.append(task)
    plan = dict(protocol=PROTOCOL, model_key=args.model, source=str(args.source.absolute()),
        styles=['neutral', 'happy'], recordings=tasks, bank=bank, prompt=gsm.ORIGINAL, prompt_variant='original-happy400',
        seed=42, max_new_tokens=512, recipe=dict(gsm.MODELS[args.model], dtype='bfloat16', do_sample=False,
            num_beams=1, use_cache=True, repetition_penalty=1., text_only_output=True),
        selection='First400 canonical1269 questions shuffled seed42; same collection cohort used for angry-neutral '
            'steering. No outcome/classifier filtering. Exploratory happy-delivery comparison, not held-out steering validation.',
        scoring='Installed upstream GSM8K filters and metric on native audio-generated responses; '
            'numeric equivalence additional; not official text-harness inference. Full audio, no steering.')
    output = args.output/args.model; output.mkdir(parents=True, exist_ok=True)
    freeze(output/'run_config.json', plan)
    return output, plan


def reuse_collection(output, plan):
    """Reuse only neutral answers with identical model, prompt, audio and decoder."""
    import torch
    from steer_minicpm_gsm import validate_pair as validate_mini
    from steer_phi4_gsm_reverse import validate_pair as validate_phi
    roots = ([Path('outputs/minicpm-gsm400-869-L30-a1')] if plan['model_key']=='minicpm' else
             [Path('outputs/phi4-gsm-reverse400-400-L30'), Path('outputs/phi4-gsm-reverse400-400-L17')])
    tasks = {(r['question_id'], r['voice']): r for r in plan['recordings'] if r['style']=='neutral'}
    done = {(r['question_id'], r['voice'], r['style']) for r in read_rows(output/'results.jsonl')}
    count = 0
    for root in roots:
        config = root/'run_config.json'
        if not config.exists(): continue
        old = json.loads(config.read_text()); config_hash = digest(config)
        if (old['prompt'] != plan['prompt'] or old['seed'] != 42 or old['max_new_tokens'] != 512
                or any(old['recipe'].get(k) != plan['recipe'].get(k) for k in plan['recipe'])):
            raise ValueError('Collection prompt/model/decoder differs; refusing baseline reuse.')
        if plan['model_key']=='minicpm':
            expected = dict(enable_thinking=False, use_tts_template=True, generate_audio=False, do_sample=False,
                max_new_tokens=512, min_new_tokens=0, num_beams=1, repetition_penalty=1., max_inp_length=32768, stream=False)
            if old['native_options'] != expected: raise ValueError('MiniCPM native options differ.')
        with (output/'results.jsonl').open('a') as stream:
            for i, pair in enumerate(old['collection']):
                key = pair['question_id'], pair['voice']; path = root/'pairs'/f'{i:04d}.pt'
                if key not in tasks or (*key, 'neutral') in done or not path.exists(): continue
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if plan['model_key']=='minicpm': validate_mini(saved, pair, config_hash)
                else: validate_phi(saved, pair, config_hash, old['protocol'])
                task = tasks[key]
                if (saved['audio_sha256']['neutral'] != task['audio_sha256']
                        or str(pair['neutral']['reference_answer']) != str(task['reference_answer'])):
                    continue  # A different synthesis needs its own neutral response.
                trace = saved['traces']['neutral']
                row = dict(task, response=trace['response'], response_tokens=trace['response_tokens'],
                    prompt=plan['prompt'], model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                    seed=42, max_new_tokens=512, interventions=[],
                    reused_evidence=dict(folder=str(root.absolute()), run_config_sha256=config_hash,
                                         pair_cache_sha256=digest(path)))
                stream.write(json.dumps(row)+'\n'); done.add((*key, 'neutral')); count += 1
    print(f'Reused{count} compatible neutral collection answers; no source results modified.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('audio', 'prepare', 'evaluate', 'summary'), default='evaluate')
    p.add_argument('--model', choices=('phi4', 'minicpm'), default='minicpm')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-gsm-happy400'))
    p.add_argument('--output', type=Path, default=Path('outputs/gsm-happy400'))
    p.add_argument('--audio-sources', nargs='+', type=Path, default=[Path('outputs')/name for name in
        ('azure-kimi-gsm8k1269', 'azure-jenny500-extra', 'azure-jenny500-neutral', 'azure-jenny-next500', 'azure-jenny-remaining-test')])
    args = p.parse_args()
    if args.mode == 'audio': audio(args); return
    if args.mode == 'summary': gsm.report(args.output/args.model); return
    output, plan = prepare(args)
    with (output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reuse_collection(output, plan)
        if args.mode == 'prepare': return
        if not candidates.pending([(output, plan)]): print('All800 answers complete; no model loaded.'); return
        with Path('outputs/jenny-task-vectors.lock').open('w') as gpu:
            print('Waiting for shared GPU lock and current models to finish.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            candidates.run([(output, plan)], args.model, 0)


if __name__ == '__main__': main()
