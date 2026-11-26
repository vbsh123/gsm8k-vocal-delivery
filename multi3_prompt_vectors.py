"""Replay both MultiArith prompts and measure voice and wording effects before decoding."""
import argparse
import csv
import io
import json
from pathlib import Path
import random

import numpy as np

from collect_diffa_reasoning_emotion import capture_initial
from diffa_model_receipts import model_revisions
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
import winning_diffa_sweeps as winners
from collect_diffa_cross_prompt import write_state

OUT = Path('outputs/diffa-multiarith-prompt-vectors270')
GROUPS = ('acoustic', 'answer-mask')


def make_plan():
    from run_multiarith_reasoning_ablation import COMPARISON, REFERENCE, SEQUENCED_PROMPT, STYLES

    source = json.loads(REFERENCE.read_text())
    original = json.loads(COMPARISON.read_text())
    if original['reference_sha256'] != winners.digest(REFERENCE):
        raise ValueError('Original and sequenced cohorts differ.')
    questions = source['questions']
    ids = list(questions)[20:]
    if len(ids) != 270:
        raise ValueError('Expected 270 questions after the pilot.')
    shuffled = ids.copy()
    random.Random(42).shuffle(shuffled)
    return dict(prompts=[original['prompt'], SEQUENCED_PROMPT], styles=list(STYLES),
                groups=list(GROUPS), layers=list(range(32)), question_order=ids,
                fold_a=shuffled[:135], recipe=source['recipe'], model='DIFFA-2',
                model_revisions=model_revisions(), reference_sha256=winners.digest(REFERENCE),
                clips={q:{s:{k:questions[q][s][k] for k in ('audio_file','audio_sha256')}
                          for s in STYLES} for q in ids})


def collect(folder, plan, pending):
    from run_multi3_replication import resolve_phi_recordings
    from run_multiarith_reasoning_ablation import STYLES
    from extract_vectors import gen_chunked, load_model
    import soundfile as sf
    import src.fast_dllm.diffa_generate as decoder

    records = [dict(question_id=q, style=s, **plan['clips'][q][s]) for q in pending for s in STYLES]
    paths = resolve_phi_recordings(records, folder)
    with gpu_queue():
        model, tokenizer, features = load_model()
        winners.freeze(folder/'runtime.json', dict(decoder_sha256=winners.digest(decoder.__file__),
                       model_revisions=model_revisions()))
        for i, q in enumerate(pending, 1):
            audio = {s:sf.read(paths[plan['clips'][q][s]['audio_sha256']], dtype='float32') for s in STYLES}
            states = []
            for prompt in plan['prompts']:
                for s in STYLES:
                    wav, rate = audio[s]
                    winners.seed()
                    trace = capture_initial(model, lambda: gen_chunked(
                        wav, rate, model, tokenizer, features, prompt, plan['recipe']['max_new_tokens'], 30), decoder)
                    # Existing hook returns acoustic, semantic, combined speech, then answer mask.
                    h = trace['states'][:, [0, 3]].numpy()
                    if h.shape[:2] != (32, 2) or not np.isfinite(h).all():
                        raise ValueError(f'Invalid state shape or values on {q}: {h.shape}')
                    states.append(h)
            write_state(folder/'states'/(q+'.npy'), np.stack(states))
            print(f'[{270-len(pending)+i}/270] {q}: four first forwards saved', flush=True)


def contrasts(folder, q):
    original_neutral, original_angry, new_neutral, new_angry = np.load(
        folder/'states'/(q+'.npy'), allow_pickle=False).astype(np.float64)
    return np.stack([original_angry-original_neutral, new_angry-new_neutral,
                     new_neutral-original_neutral, new_angry-original_angry])


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.full_like(x, np.nan), where=norm>0)


def report(folder, plan):
    planned = plan['question_order']
    # Each state file is renamed into place only after all four forwards are saved.
    ids = [q for q in planned if (folder/'states'/(q+'.npy')).is_file()]
    n = len(ids)
    partial = n < len(planned)
    first = set(plan['fold_a'])
    count_a = sum(q in first for q in ids)
    count_b = n - count_a
    print(f'{"Interim" if partial else "Final"}: {n}/{len(planned)} matched questions '
          f'({4*n} initial forwards). Folds A/B: {count_a}/{count_b}.', flush=True)
    if not count_a or not count_b:
        print('Cross-fold estimates need at least one saved question in each fold.')
        return
    sums = [None, None]
    per_item_norms = []
    for q in ids:
        d = contrasts(folder, q)
        f = int(q not in first)
        if sums[f] is None:
            sums[f] = np.zeros_like(d)
        sums[f] += d
        per_item_norms.append(np.linalg.norm(d, axis=-1))
    means = [sums[0]/count_a, sums[1]/count_b]
    axes = [unit(m) for m in means]
    def cosine(f,i,g,j):
        return (axes[f][i]*axes[g][j]).sum(-1)
    result = {}
    for name,i,j in [('voice_shared_across_prompts',0,1),
                     ('neutral_rewrite_vs_original_voice',2,0), ('angry_rewrite_vs_original_voice',3,0),
                     ('rewrite_shared_across_voices',2,3)]:
        result[name+'_A_vs_B'] = cosine(0,i,1,j)
        result[name+'_B_vs_A'] = cosine(1,i,0,j)
    avg_norm = np.mean(per_item_norms, axis=0)
    for i,name in enumerate(('original_voice','sequenced_voice','neutral_rewrite','angry_rewrite')):
        result[name+'_split_half_cosine'] = cosine(0,i,1,i)
        result[name+'_mean_question_norm'] = avg_norm[i]
        result[name+'_mean_vector_norm'] = np.linalg.norm((sums[0][i]+sums[1][i])/n,axis=-1)
    # For each question, estimate its displacement along the original voice axis
    # using only other questions to define the axis. Positive means toward original angry.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['question_id','fold','layer','token_group','neutral_rewrite_projection','angry_rewrite_projection'])
    projection_sum = np.zeros_like(avg_norm[2:])
    for q in ids:
        f = int(q not in first)
        d = contrasts(folder,q)
        projections = (d[2:]*axes[1-f][0]).sum(-1)
        projection_sum += projections
        for layer in plan['layers']:
            for j, group in enumerate(GROUPS):
                writer.writerow([q,'AB'[f],layer,group,*projections[:,layer,j]])
    result['neutral_rewrite_mean_projection'], result['angry_rewrite_mean_projection'] = projection_sum / n
    prefix = 'interim-' if partial else ''
    question_csv = folder / (prefix + 'question-projections.csv')
    layer_csv = folder / (prefix + 'layer-comparison.csv')
    winners.atomic_text(question_csv,buf.getvalue())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['layer','token_group',*result])
    for layer in plan['layers']:
        for j, group in enumerate(GROUPS):
            writer.writerow([layer,group,*[x[layer,j] for x in result.values()]])
    winners.atomic_text(layer_csv,buf.getvalue())
    print('Shift = mean per-question distance from original to reworded prompt, with audio fixed.')
    print('Projection = mean signed shift along the original angry-minus-neutral direction from the other fold.')
    print('Shared cosine compares the angry-minus-neutral directions across prompts in opposite folds.')
    print(f'{"Layer":>5}  {"Tokens":<11} {"Neutral shift":>13} {"Angry shift":>12} '
          f'{"Neutral proj":>13} {"Angry proj":>11}  Shared cosine A/B, B/A')
    for layer in plan['layers']:
        for j, group in enumerate(GROUPS):
            metrics = [result[key][layer,j] for key in (
                'neutral_rewrite_mean_question_norm','angry_rewrite_mean_question_norm',
                'neutral_rewrite_mean_projection','angry_rewrite_mean_projection',
                'voice_shared_across_prompts_A_vs_B','voice_shared_across_prompts_B_vs_A')]
            ns, ang, npj, apj, ab, ba = metrics
            print(f'{layer:5d}  {group:<11} {ns:13.3f} {ang:12.3f} {npj:+13.3f} '
                  f'{apj:+11.3f}  {ab:+.4f}, {ba:+.4f}')
    print(f'All {len(plan["layers"])} layers: {layer_csv}\nPer-question shifts: {question_csv}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run','summary'])
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    folder = args.output
    if args.mode == 'summary':
        config = folder/'run_config.json'
        if config.is_file():
            report(folder,json.loads(config.read_text()))
        else:
            print('Awaiting collection.')
        return
    with job_lock(folder):
        plan = make_plan()
        winners.freeze(folder/'run_config.json',plan)
        (folder/'states').mkdir(exist_ok=True)
        pending = [q for q in plan['question_order'] if not (folder/'states'/(q+'.npy')).exists()]
        if pending:
            collect(folder,plan,pending)
        report(folder,plan)


if __name__ == '__main__':
    main()
