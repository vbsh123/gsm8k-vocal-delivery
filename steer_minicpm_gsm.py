"""MiniCPM native GSM steering: collect400 angry-neutral means, test869 at L30+1."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path

import diagnose_minicpm_gsm100 as diagnostics
import evaluate_gsm_candidates as gsm
import steer_phi4_gsm_reverse as engine
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze

PROTOCOL = 'minicpm-gsm400-test869-L30-answer-a1-native-mean-v1'
DEFAULT = Path('outputs/minicpm-gsm400-869-L30-a1')


def prepare(args):
    path = args.output/'run_config.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if (plan['protocol'] != PROTOCOL or plan['source'] != str(args.source.absolute())
                or plan['gpu_budget_gib'] != args.gpu_budget_gib or plan['recipe']['revision'] != diagnostics.SPEC['revision']):
            raise ValueError('Frozen MiniCPM settings changed; use another output.')
        return plan
    parent_path = args.diagnostics/'run_config.json'
    prior = json.loads(parent_path.read_text())
    if prior['protocol'] != diagnostics.PROTOCOL or prior['recipe'] != diagnostics.SPEC:
        raise ValueError('Need matching pinned MiniCPM diagnostic config.')
    source = gsm.make_plan(args.source, 'minicpm', 'original')
    if prior['reasoning_prompt'] != source['prompt']:
        raise ValueError('Prompt differs from inspected MiniCPM diagnostics.')
    questions = gsm.questions()
    if len(questions) != 1269: raise ValueError('Expected1269 questions.')
    records = {(r['question_id'], r['style']): r for r in source['recordings']}
    collection = []
    for q in questions[:400]:
        neutral, angry = (records[q['question_id'], s] for s in ('neutral', 'angry'))
        if neutral['voice'] != angry['voice']: raise ValueError('Voices differ.')
        collection.append(dict(question_id=q['question_id'], voice=neutral['voice'], neutral=neutral, angry=angry))
    test = [records[q['question_id'], 'neutral'] for q in questions[400:]]
    if {p['question_id'] for p in collection}&{r['question_id'] for r in test}: raise ValueError('Split overlap.')
    plan = dict(protocol=PROTOCOL, source=str(args.source.absolute()), recipe=source['recipe'],
                diagnostic_config=str(parent_path.absolute()), diagnostic_config_sha256=digest(parent_path),
                question_file_sha256=digest(gsm.QUESTION_FILE), seed=42, max_new_tokens=512,
                prompt=prior['reasoning_prompt'], native_options=prior['native_options'],
                layer=30, group='answer', strengths={'neutral': 0., 'neutral-L30-answer-a1': 1.},
                collection=collection, test=test, bank={q['question_id']: q for q in questions[400:]},
                gpu_budget_gib=args.gpu_budget_gib, report_label='MiniCPM GSM8K own-vector',
                vector_definition='Equal mean of400 angry-minus-neutral per-question differences. Each style uses '
                    'the equal mean of last answer-prediction states across ALL native generation forwards, '
                    'including prefill. Own generated histories, no teacher forcing, no normalization. '
                    'Same positions are injected on every forward. Not the initial-state diagnostic vector.',
                selection='Canonical1269 shuffled seed42; first400 collection, rest869 test. No outcome/classifier '
                    'filter. L30/+1 chosen after inspecting SAME100 diagnostics; those100 may overlap the869 '
                    'test questions, so this is exploratory, not untouched confirmatory validation.',
                scoring='Installed upstream GSM8K task filters and exact-match; numeric equivalence additional. '
                    'Native spoken adaptation, not official text-harness inference. Paired tests/intervals unadjusted.')
    freeze(path, plan)
    print('Frozen400 collection pairs +869 disjoint tests; L30 answer+1; existing full audio.', flush=True)
    return plan


class NativeMini:
    def __init__(self, plan):
        import torch
        from transformers import AutoModel, AutoProcessor
        if version('transformers') != diagnostics.SPEC['transformers']: raise ValueError('Use .venv-bbh-minicpm.')
        path = diagnostics.checkpoint('minicpm')
        diagnostics.cap_gpu(plan['gpu_budget_gib'])
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', init_vision=False, init_audio=True, init_tts=False).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        layers = self.model.llm.model.layers
        if len(layers) != 36 or self.model.llm.config.hidden_size != 4096:
            raise ValueError('Pinned MiniCPM decoder layout differs.')
        self.layer = layers[plan['layer']]
        self.options = plan['native_options']

    def generate(self, row, prompt, direction=None, strength=0., observe=False):
        import torch
        if digest(row['audio_file']) != row['audio_sha256']: raise ValueError('Audio changed.')
        engine.seed()
        with torch.inference_mode(), engine.answer_hook(self.layer, direction, strength, observe) as capture:
            response = self.model.chat(msgs=[{'role': 'user', 'content': [prompt, diagnostics.load_audio(row['audio_file'])]}],
                                       processor=self.processor, **self.options)
        if not isinstance(response, str): raise ValueError('Expected native text output.')
        result = dict(response=response, response_tokens=len(self.processor.tokenizer.encode(response, add_special_tokens=False)),
                      hook_forwards=capture['count'])
        if observe:
            if capture['count'] <= 0: raise ValueError('No native decoder forwards.')
            result['mean'] = capture['total']/capture['count']
        return result


def validate_pair(saved, pair, config_hash):
    import torch
    if (saved['protocol'] != PROTOCOL or saved['run_config_sha256'] != config_hash
            or saved['question_id'] != pair['question_id'] or saved['voice'] != pair['voice']
            or saved['audio_sha256'] != {s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}):
        raise ValueError('Cached MiniCPM collection metadata differs.')
    for trace in saved['traces'].values():
        if trace['hook_forwards'] <= 0 or trace['mean'].shape != (4096,) or not torch.isfinite(trace['mean']).all():
            raise ValueError('Invalid native collection mean.')
    expected = saved['traces']['angry']['mean']-saved['traces']['neutral']['mean']
    if not torch.equal(saved['answer_difference'], expected): raise ValueError('Cached difference sign/value differs.')


def collect(plan, output, native, config_hash):
    import torch
    folder = output/'pairs'; folder.mkdir(exist_ok=True)
    differences = []
    for i, pair in enumerate(plan['collection']):
        path = folder/f'{i:04d}.pt'
        if path.exists(): saved = torch.load(path, map_location='cpu', weights_only=True)
        else:
            print(f'Collect [{i+1}/400] {pair["question_id"]}: neutral+angry L30.', flush=True)
            traces = {s: native.generate(pair[s], plan['prompt'], observe=True) for s in ('neutral', 'angry')}
            saved = dict(protocol=PROTOCOL, run_config_sha256=config_hash, question_id=pair['question_id'], voice=pair['voice'],
                         audio_sha256={s: pair[s]['audio_sha256'] for s in traces}, traces=traces,
                         answer_difference=traces['angry']['mean']-traces['neutral']['mean'])
            validate_pair(saved, pair, config_hash); atomic_save(path, saved)
        validate_pair(saved, pair, config_hash); differences.append(saved['answer_difference'])
    vector = dict(protocol=PROTOCOL, run_config_sha256=config_hash, model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                  layer=30, group='answer', definition=plan['vector_definition'],
                  collection_question_ids=[p['question_id'] for p in plan['collection']],
                  angry_minus_neutral=torch.stack(differences).mean(0))
    path = output/'style-vectors.pt'
    if path.exists():
        old = torch.load(path, map_location='cpu', weights_only=True)
        if old['run_config_sha256'] != config_hash or not torch.equal(old['angry_minus_neutral'], vector['angry_minus_neutral']):
            raise ValueError('Existing MiniCPM vector differs.')
    else: atomic_save(path, vector)
    return vector['angry_minus_neutral']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'run', 'summary'), default='run')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-gsm8k1269'))
    p.add_argument('--diagnostics', type=Path, default=Path('outputs/minicpm-gsm100-diagnostics'))
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--gpu-budget-gib', type=float, default=22.)
    args = p.parse_args()
    if args.mode == 'summary': engine.report(args.output); return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        if args.mode == 'prepare': return
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        config_hash = digest(args.output/'run_config.json')
        _, done = engine.completed(plan, args.output, config_hash, repair=True)
        if len(done) == 2*len(plan['test']): print('All1738 test answers complete; no model loaded.'); return
        print('Concurrent MiniCPM: no shared GPU lock or Phi queue wait;22 GiB Torch cap.', flush=True)
        native = NativeMini(plan)
        (args.output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, transformers=version('transformers'),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), script_sha256=digest(__file__),
            engine_sha256=digest(engine.__file__), native_options=plan['native_options'],
            gpu_budget_gib=plan['gpu_budget_gib']), indent=2)+'\n')
        vector = collect(plan, args.output, native, config_hash)
        engine.evaluate(plan, args.output, native, vector, config_hash)


if __name__ == '__main__': main()
