"""Calibrate SAME100 emotion/reasoning geometry against other instruction directions."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from audit_reasoning_emotion import CONDITIONS, PROTOCOLS, geometry, state_shape

PROTOCOL = 'same100-instruction-controls-initial-v1'
# These replace only the reasoning sentence; the task and final-answer suffix remain fixed.
INSTRUCTIONS = {
    'reason_paraphrase': 'Work through the necessary calculations step by step before giving the final answer.',
    'reason_paraphrase2': 'Derive the answer through a sequence of intermediate steps.',
    'concise': 'Keep your response concise.',
    'detailed': 'Explain your solution in detail.',
    'format': 'Follow the requested final-answer format carefully.',
    'verify': 'Check your calculations before giving the final answer.',
    'careful': 'Pay close attention to every detail in the question.',
    'confident': 'Respond confidently.',
    'polite': 'Respond politely.',
    'no_extra_instruction': '',
}
REASON_PHRASES = ("Let's think step by step.",
                  'Give concise step-by-step reasoning with only the necessary calculations.',
                  'Give concise step-by-step reasoning with only the necessary deductions.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def prompts(original):
    matches = [s for s in REASON_PHRASES if s in original]
    if len(matches) != 1 or original.count(matches[0]) != 1:
        raise ValueError('Expected exactly one known reasoning sentence.')
    phrase = matches[0]
    return {k: original.replace(phrase, text) for k, text in INSTRUCTIONS.items()}


def state_path(folder, index, condition):
    return Path(folder)/'states'/f'{index:03d}-{condition}.pt'


def validate(trace, parent, pair, condition, prompt, protocol, model):
    import torch
    clip = pair['angry' if condition == 'angry-reasoning' else 'neutral']
    expected = dict(protocol=protocol, question_id=pair['question_id'], voice=pair['voice'],
                    condition=condition, prompt=prompt, audio_sha256=clip['audio_sha256'])
    if any(trace.get(k) != v for k, v in expected.items()):
        raise ValueError('State metadata differs from frozen question/audio/instruction.')
    h = trace['states'] if model == 'diffa' else torch.stack((trace['answer'], trace['audio']), dim=1)
    if tuple(h.shape) != state_shape(model) or not torch.isfinite(h).all():
        raise ValueError('Invalid/nonfinite state tensor.')
    return h.double()


def prepare(args):
    import torch
    config = args.output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if (plan['protocol'] != PROTOCOL or plan['model_kind'] != args.model
                or plan['parent_folder'] != str(args.parent.absolute())):
            raise ValueError('Different settings; use another output folder.')
        verify_parent(plan)
        print('Reusing EXACT same frozen100 questions and control prompts.', flush=True)
        return plan
    parent_path = args.parent/'run_config.json'; parent = json.loads(parent_path.read_text())
    if (parent['protocol'] != PROTOCOLS[args.model] or len(parent['pairs']) != 100
            or len({(p['question_id'], p['voice']) for p in parent['pairs']}) != 100):
        raise ValueError('Need an existing complete SAME100 collector, without reselection.')
    receipts = []
    for i, pair in enumerate(parent['pairs']):
        for condition in CONDITIONS:
            path = state_path(args.parent, i, condition)
            prompt = parent['answer_only_prompt'] if condition == 'neutral-answer-only' else parent['reasoning_prompt']
            validate(torch.load(path, map_location='cpu', weights_only=True), parent, pair,
                     condition, prompt, parent['protocol'], args.model)
            receipts.append(dict(path=str(path.absolute()), sha256=digest(path)))
    plan = dict(protocol=PROTOCOL, model_kind=args.model, parent_folder=str(args.parent.absolute()),
                parent_config_sha256=digest(parent_path), parent=parent, source_states=receipts,
                prompts=prompts(parent['reasoning_prompt']), new_forwards=100*len(INSTRUCTIONS),
                reference='Existing neutral-answer-only states; alternate no-extra-instruction reference also reported.',
                emotion='Existing angry-reasoning minus neutral-reasoning; unchanged in both reference analyses.',
                selection='EXACT original100 pairs in original order; no selection or outcome filtering.',
                limitations='Instruction-request geometry before generation; not actual reasoning or causal accuracy. '
                'Prompt wording/length and stable response-style effects remain possible. Default model cohorts/capture sites differ.')
    config.write_text(json.dumps(plan, indent=2)+'\n')
    print(f'Frozen SAME100; {len(INSTRUCTIONS)} new instructions = {plan["new_forwards"]} initial forwards. '
          'Original reasoning/emotion states reused; no TTS or full generations.', flush=True)
    return plan


def verify_parent(plan):
    folder = Path(plan['parent_folder'])
    if digest(folder/'run_config.json') != plan['parent_config_sha256']:
        raise ValueError('Parent config changed after controls were frozen.')
    for receipt in plan['source_states']:
        if digest(receipt['path']) != receipt['sha256']:
            raise ValueError('Parent state changed after controls were frozen.')


def compare_replay(actual, expected):
    """Catch model/runtime drift before comparing new states to a saved reference."""
    import torch
    actual, expected = actual.double(), expected.double()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError('Native replay has invalid/nonfinite states.')
    error = float((actual-expected).abs().max())
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        raise ValueError(f'Native reference replay differs from parent states (max error {error:.6g}). '
                         'Check model weights/runtime before comparing directions; no controls collected.')
    return dict(max_absolute_error=error, atol=1e-4, rtol=1e-4, status='PASS')


def collect(plan, output):
    import numpy as np
    import torch
    model_kind = plan['model_kind']; parent = plan['parent']; pending = []
    for i, pair in enumerate(parent['pairs']):
        for condition, prompt in plan['prompts'].items():
            path = state_path(output, i, condition)
            if path.exists():
                validate(torch.load(path, map_location='cpu', weights_only=True), parent, pair,
                         condition, prompt, PROTOCOL, model_kind)
            else: pending.append((i, pair, condition, prompt, path))
    print(f'{len(pending)}/{plan["new_forwards"]} instruction-control forwards pending.', flush=True)
    if not pending: return
    if model_kind == 'diffa':
        import soundfile as sf
        import src.fast_dllm.diffa_generate as decoder
        from extract_vectors import load_model, gen_chunked
        from collect_diffa_reasoning_emotion import capture_initial
        model, tokenizer, features = load_model()
    else:
        from importlib.metadata import version
        from transformers import AutoModelForCausalLM, AutoProcessor
        from bbh_candidate_adapters import MODELS, checkpoint, load_audio, phi_prompt
        from collect_phi4_reasoning_emotion import capture
        if version('transformers') != parent['transformers']:
            raise ValueError('Use the original .venv-bbh-phi4 environment.')
        checkpoint_path = checkpoint('phi4')
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map='cuda:0', _attn_implementation='sdpa').eval()
    from importlib.metadata import version
    runtime = dict(torch=torch.__version__, transformers=version('transformers'), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), collector_sha256=digest(__file__), model_kind=model_kind,
                   parent_runtime_sha256=digest(Path(plan['parent_folder'])/'runtime.json'),
                   capture='Unchanged original native collector function; first forward only.')
    if model_kind == 'diffa':
        runtime.update(decoder_sha256=digest(decoder.__file__), model_config=model.config.to_dict())
    (output/'runtime.json').write_text(json.dumps(runtime, indent=2, default=str)+'\n')
    def capture_clip(clip, prompt):
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        if model_kind == 'diffa':
            audio, rate = sf.read(clip['audio_file'], dtype='float32')
            return capture_initial(model, lambda: gen_chunked(audio, rate, model, tokenizer, features, prompt, 512, 30), decoder)
        inputs = processor(text=phi_prompt(prompt), audios=[(load_audio(clip['audio_file']), 16000)],
                           return_tensors='pt').to(model.device, dtype=torch.bfloat16)
        return capture(model, inputs)
    # One previously collected neutral forward detects checkpoint/runtime drift.
    pair = parent['pairs'][0]; clip = pair['neutral']
    if digest(clip['audio_file']) != clip['audio_sha256']: raise ValueError('Replay audio changed.')
    print('Replaying one original neutral reasoning forward to verify the saved reference.', flush=True)
    replay = capture_clip(clip, parent['reasoning_prompt'])
    expected = torch.load(state_path(plan['parent_folder'], 0, 'neutral-reasoning'), map_location='cpu', weights_only=True)
    actual_h = replay['states'] if model_kind == 'diffa' else torch.stack((replay['answer'], replay['audio']), dim=1)
    expected_h = expected['states'] if model_kind == 'diffa' else torch.stack((expected['answer'], expected['audio']), dim=1)
    replay_report = compare_replay(actual_h, expected_h)
    (output/'reference-replay.json').write_text(json.dumps(dict(question_id=pair['question_id'], **replay_report), indent=2)+'\n')
    (output/'states').mkdir(exist_ok=True)
    for count, (i, pair, condition, prompt, path) in enumerate(pending, 1):
        clip = pair['neutral']
        if digest(clip['audio_file']) != clip['audio_sha256']: raise ValueError('Neutral audio changed.')
        print(f'Control [{count}/{len(pending)}] {pair["question_id"]} / {condition}', flush=True)
        trace = capture_clip(clip, prompt)
        if model_kind == 'diffa':
            baseline = torch.load(state_path(plan['parent_folder'], i, 'neutral-reasoning'), map_location='cpu', weights_only=True)
            if (trace['answer_input_sha256'] != baseline['answer_input_sha256']
                    or trace['speech_bounds'] != baseline['speech_bounds']):
                raise ValueError('Control changed native answer-mask embeddings or neutral audio boundaries.')
        saved = dict(protocol=PROTOCOL, question_id=pair['question_id'], voice=pair['voice'], condition=condition,
                     prompt=prompt, audio_sha256=clip['audio_sha256'], **trace)
        validate(saved, parent, pair, condition, prompt, PROTOCOL, model_kind)
        tmp = path.with_suffix('.tmp'); torch.save(saved, tmp); tmp.replace(path)


def summarize(plan, output):
    import torch
    parent = plan['parent']; model = plan['model_kind']; nr, na, emotion = [], [], []
    for i, pair in enumerate(parent['pairs']):
        hs = []
        for c in CONDITIONS:
            prompt = parent['answer_only_prompt'] if c == 'neutral-answer-only' else parent['reasoning_prompt']
            trace = torch.load(state_path(plan['parent_folder'], i, c), map_location='cpu', weights_only=True)
            hs.append(validate(trace, parent, pair, c, prompt, parent['protocol'], model))
        nr.append(hs[0]); na.append(hs[1]); emotion.append(hs[2]-hs[0])
    nr, na, emotion = [torch.stack(a) for a in (nr, na, emotion)]
    em = emotion.mean(0)
    groups = parent['groups']; vectors = {}; complete = []
    for condition, prompt in plan['prompts'].items():
        if not all(state_path(output, i, condition).exists() for i in range(100)): continue
        hs = []
        for i, pair in enumerate(parent['pairs']):
            trace = torch.load(state_path(output, i, condition), map_location='cpu', weights_only=True)
            hs.append(validate(trace, parent, pair, condition, prompt, PROTOCOL, model))
        vectors[condition] = torch.stack(hs); complete.append(condition)
    rows = []
    references = {'answer_only': na}
    if 'no_extra_instruction' in vectors: references['bare_task'] = vectors['no_extra_instruction']
    all_states = dict(original_reasoning=nr, answer_only=na, **vectors)
    print(f'{model}: {len(complete)}/{len(plan["prompts"])} control conditions complete on ALL100.', flush=True)
    print('Reference    Instruction                 L26 answer cosine   L31 answer cosine')
    for reference, baseline in references.items():
        for condition, states in all_states.items():
            if (reference == 'answer_only' and condition == 'answer_only') or (reference == 'bare_task' and condition == 'no_extra_instruction'): continue
            deltas = states-baseline; mean = deltas.mean(0); shown = {}
            for g, group in enumerate(groups):
                for layer in range(32):
                    geo = geometry(mean[layer, g], em[layer, g])
                    c0 = geometry(deltas[::2, layer, g].mean(0), emotion[1::2, layer, g].mean(0))['cosine']
                    c1 = geometry(deltas[1::2, layer, g].mean(0), emotion[::2, layer, g].mean(0))['cosine']
                    rows.append(dict(reference=reference, instruction=condition, group=group, layer=layer, pairs=100,
                                     cosine=geo['cosine'], signed_projection_coefficient=geo['coefficient'],
                                     squared_norm_overlap=geo['overlap'], instruction_norm=float(mean[layer, g].norm()),
                                     emotion_norm=float(em[layer, g].norm()), cross_half_A_B=c0, cross_half_B_A=c1))
                    if group == 'answer' and layer in (26, 31):
                        shown[layer] = 'undefined' if geo['cosine'] is None else f'{geo["cosine"]:+.4f}'
            print(f'{reference:12} {condition:26} {shown[26]:>17} {shown[31]:>19}', flush=True)
    with (output/'instruction-cosines.csv').open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL, complete_controls=complete,
        model_kind=model, rows=rows, limitations=plan['limitations'], reference=plan['reference']), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=PROTOCOLS, required=True)
    p.add_argument('--mode', choices=('prepare', 'collect', 'summary'), default='collect')
    p.add_argument('--parent', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.parent = args.parent or Path('outputs')/('diffa-reasoning-emotion100' if args.model == 'diffa' else 'phi4-reasoning-emotion100-gsm8k')
    args.output = args.output or Path('outputs')/f'{args.model}-instruction-controls100'
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        if args.mode == 'prepare': return
        if args.mode == 'collect':
            with Path('outputs/jenny-task-vectors.lock').open('w') as gpu:
                print('Waiting for shared GPU lock.', flush=True); fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    print('Waiting for active GPU work.', flush=True); time.sleep(30)
                collect(plan, args.output)
        summarize(plan, args.output)


if __name__ == '__main__': main()
