"""CPU-only independent checks of saved SAME100 states, vectors and CSVs.

Does not load a model, rerun forwards, or change existing results.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ('neutral-reasoning', 'neutral-answer-only', 'angry-reasoning')
PROTOCOLS = {'diffa': 'diffa-same100-reasoning-emotion-initial-mask-v1',
             'phi4': 'phi4-same100-reasoning-emotion-prefill-v1'}


def state_shape(model):
    return (32, 4, 4096) if model == 'diffa' else (32, 2, 3072)


def geometry(reason, emotion):
    """Python float64/fsum arithmetic, independent of collector torch.dot/norm."""
    a, b = reason.tolist(), emotion.tolist()
    aa = math.fsum(x*x for x in a)
    bb = math.fsum(x*x for x in b)
    ab = math.fsum(x*y for x, y in zip(a, b))
    if aa == 0 or bb == 0:
        return dict(cosine=None, coefficient=None if aa == 0 else ab/aa,
                    overlap=None, orthogonality=None, pythagorean_error=None)
    coefficient = ab/aa
    projection = [coefficient*x for x in a]
    residual = [y-p for y, p in zip(b, projection)]
    pp = math.fsum(x*x for x in projection)
    rr = math.fsum(x*x for x in residual)
    return dict(cosine=ab/math.sqrt(aa*bb), coefficient=coefficient, overlap=pp/bb,
                orthogonality=abs(math.fsum(x*y for x, y in zip(a, residual)))/math.sqrt(aa*bb),
                pythagorean_error=abs(pp+rr-bb)/bb)


def audit(output, model, check_audio=False):
    import torch
    plan = json.loads((output/'run_config.json').read_text())
    errors, warnings, checks = [], [], []
    if plan['protocol'] != PROTOCOLS[model]:
        raise ValueError('Wrong collector protocol/model for this folder.')
    if len(plan['pairs']) != 100 or len({(p['question_id'], p['voice']) for p in plan['pairs']}) != 100:
        raise ValueError('Expected 100 unique frozen question/voice pairs.')
    if plan['conditions'] != list(CONDITIONS):
        raise ValueError('Unexpected three-condition order.')
    groups = ['acoustic', 'semantic', 'combined', 'answer'] if model == 'diffa' else ['answer', 'audio']
    if plan['groups'] != groups:
        raise ValueError('Unexpected token-group order.')
    reason, emotion, lengths = [], [], []
    shape = state_shape(model)
    for i, pair in enumerate(plan['pairs']):
        traces = []
        for condition in CONDITIONS:
            path = output/'states'/f'{i:03d}-{condition}.pt'
            trace = torch.load(path, map_location='cpu', weights_only=True)
            clip = pair['angry' if condition == 'angry-reasoning' else 'neutral']
            expected = dict(protocol=plan['protocol'], question_id=pair['question_id'], voice=pair['voice'],
                            condition=condition, audio_sha256=clip['audio_sha256'],
                            prompt=plan['answer_only_prompt'] if condition == 'neutral-answer-only' else plan['reasoning_prompt'])
            if any(trace.get(k) != v for k, v in expected.items()):
                raise ValueError(f'{path}: metadata differs from frozen prompt/audio/question.')
            h = trace['states'] if model == 'diffa' else torch.stack((trace['answer'], trace['audio']), dim=1)
            if tuple(h.shape) != shape or not h.is_floating_point() or not torch.isfinite(h).all():
                raise ValueError(f'{path}: invalid shape/dtype or nonfinite state.')
            traces.append((trace, h.double()))
        nr, na, ar = [h for _, h in traces]
        reason.append(nr-na)
        emotion.append(ar-nr)
        metadata = [t for t, _ in traces]
        if model == 'diffa':
            if len({t['answer_input_sha256'] for t in metadata}) != 1:
                raise ValueError('Answer-mask input embeddings differ across conditions.')
            for t in metadata:
                b = t['speech_bounds']
                if (t['answer_tokens'] != 512 or not 0 <= b['start'] < b['acoustic_end'] < b['end'] <= t['input_tokens']-512):
                    raise ValueError('Invalid speech/answer token boundaries.')
            if metadata[0]['speech_bounds'] != metadata[1]['speech_bounds']:
                raise ValueError('Same neutral audio has different token boundaries across prompts.')
            # Both means must describe the same concatenated speech span.
            for t, h in traces:
                b = t['speech_bounds']; n = b['acoustic_end']-b['start']; m = b['end']-b['acoustic_end']
                if not torch.allclose(h[:, 2], (n*h[:, 0]+m*h[:, 1])/(n+m), atol=2e-5, rtol=2e-5):
                    raise ValueError('Combined audio mean disagrees with acoustic/semantic means.')
        else:
            if metadata[0]['audio_tokens'] != metadata[1]['audio_tokens']:
                raise ValueError('Neutral audio token count changed across text prompts.')
        lengths.append(dict(question_id=pair['question_id'],
                            input_tokens={c: t['input_tokens'] for c, t in zip(CONDITIONS, metadata)}))
        if check_audio:
            for style in ('neutral', 'angry'):
                clip = pair[style]
                if hashlib.sha256(Path(clip['audio_file']).read_bytes()).hexdigest() != clip['audio_sha256']:
                    raise ValueError('Audio file no longer matches collection hash.')
    r, e = torch.stack(reason), torch.stack(emotion)
    rm, em = r.mean(0), e.mean(0)
    vectors = torch.load(output/'style-vectors.pt', map_location='cpu', weights_only=True)
    expected_vectors = {'reasoning_minus_answer_only': rm if model == 'diffa' else rm[:, 0],
                        'angry_minus_neutral': em if model == 'diffa' else em[:, 0],
                        'neutral_minus_angry': -em if model == 'diffa' else -em[:, 0]}
    if model == 'phi4':
        expected_vectors.update(audio_reasoning_minus_answer_only=rm[:, 1], audio_angry_minus_neutral=em[:, 1],
                                audio_neutral_minus_angry=-em[:, 1])
    if vectors.get('protocol') != plan['protocol'] or vectors.get('pairs') != 100:
        errors.append('Vector file protocol/pair count differs from plan.')
    for key, value in expected_vectors.items():
        actual = vectors[key].double()
        if actual.shape != value.shape or not torch.allclose(actual, value, atol=2e-5, rtol=2e-5):
            errors.append(f'Saved vector differs from raw states: {key}')
    with (output/'layer-cosines.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    summary = json.loads((output/'summary.json').read_text())
    if summary['pairs'] != 100 or summary['protocol'] != plan['protocol']:
        errors.append('Summary protocol/pair count differs from plan.')
    summary_rows = {(r.get('group', 'answer'), r['layer']): r for r in summary['layers']}
    expected_keys = {(g, l) for g in (groups if model == 'diffa' else ['answer']) for l in range(32)}
    keys = [(row.get('group', 'answer'), int(row['layer'])) for row in csv_rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        errors.append('CSV has missing/duplicate/wrong group-layer rows.')
    if set(summary_rows) != expected_keys or len(summary['layers']) != len(expected_keys):
        errors.append('Summary has missing/duplicate/wrong group-layer rows.')
    max_error = 0.
    for row in csv_rows:
        group, layer = row.get('group', 'answer'), int(row['layer']); g = groups.index(group)
        geo = geometry(rm[layer, g], em[layer, g])
        if geo['orthogonality'] is not None and (geo['orthogonality'] > 1e-10 or geo['pythagorean_error'] > 1e-10):
            errors.append(f'Projection identities failed: {group}/{layer}')
        r0, r1 = r[::2, layer, g].mean(0), r[1::2, layer, g].mean(0)
        e0, e1 = e[::2, layer, g].mean(0), e[1::2, layer, g].mean(0)
        expected = dict(cosine_angry_neutral_vs_reasoning=geo['cosine'],
                        cosine_neutral_angry_vs_reasoning=None if geo['cosine'] is None else -geo['cosine'],
                        signed_projection_coefficient=geo['coefficient'], squared_norm_overlap=geo['overlap'],
                        reasoning_norm=float(rm[layer, g].norm()), emotion_norm=float(em[layer, g].norm()),
                        reasoning_split_half=geometry(r0, r1)['cosine'], emotion_split_half=geometry(e0, e1)['cosine'],
                        cross_half_reason_even_emotion_odd=geometry(r0, e1)['cosine'],
                        cross_half_reason_odd_emotion_even=geometry(r1, e0)['cosine'])
        for key, value in expected.items():
            saved = float(row[key]) if row.get(key) else None
            json_value = summary_rows.get((group, layer), {}).get(key)
            if saved != json_value:
                errors.append(f'CSV and summary disagree: {group}/{layer}/{key}')
            if saved is not None and value is not None:
                max_error = max(max_error, abs(saved-value))
            if ((saved is None) != (value is None) or
                    (value is not None and not math.isclose(saved, value, abs_tol=2e-5, rel_tol=2e-5))):
                errors.append(f'CSV mismatch: {group}/{layer}/{key}: saved={saved}, recomputed={value}')
        checks.append(dict(group=group, layer=layer, **geo))
    warnings.extend([
        'This checks saved-state arithmetic/metadata, not whether a representation is actual reasoning.',
        'Instruction wording and token positions change; angry and neutral audio can have different token lengths.',
        'DIFFA averages 512 initial masks; Phi uses the last prompt position. Cross-model magnitudes are not matched measurements.',
        'Cross-half checks remove shared question noise, not common prompt/style/position effects.',
        'Default DIFFA and Phi reasoning wording/cohort selection differ; opposite signs are not a controlled model-only comparison.',
        'No model forward was rerun. Native runtime equivalence and checkpoint identity require collection-time provenance.'
    ])
    report = dict(model=model, pairs=100, status='FAIL' if errors else 'PASS', errors=errors,
                  reasoning_prompt=plan['reasoning_prompt'], answer_only_prompt=plan['answer_only_prompt'],
                  question_ids=[p['question_id'] for p in plan['pairs']],
                  max_csv_absolute_error=max_error, checks=checks, input_lengths=lengths, limitations=warnings)
    # Separate audit artifact: original states/vectors/summaries remain unchanged.
    (output/'independent-geometry-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=PROTOCOLS, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-audio', action='store_true', help='Also hash original WAV files; requires them locally.')
    args = p.parse_args()
    output = args.output or Path('outputs')/('diffa-reasoning-emotion100' if args.model == 'diffa' else 'phi4-reasoning-emotion100-gsm8k')
    result = audit(output, args.model, args.check_audio)
    print(f"{result['status']}: {args.model}; 100 triples; independent float64 arithmetic; max CSV error={result['max_csv_absolute_error']:.3g}")
    for error in result['errors']: print('ERROR:', error)
    print('Saved:', output/'independent-geometry-audit.json')
    raise SystemExit(1 if result['errors'] else 0)


if __name__ == '__main__': main()
