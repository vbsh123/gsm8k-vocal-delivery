"""Phi GSM8K neutral/angry bare-prompt evaluation only; existing Jenny audio, all1319."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import time

from bbh_candidate_adapters import MODELS
from jenny_bundle_registry import Registry, sha
from steer import read_rows
import winning_diffa_sweeps as winners

PROTOCOL = 'phi4-gsm-bare-neutral-angry-eval-only-v1'
PROMPT = ('Solve the math problem in the audio. End your response with a final answer line in the format:\n'
          '#### <number>')
STYLES = ('neutral', 'angry')
TOTAL = 1319
PARENT = Path('outputs/emotion-block-night/phi/gsm8k/run_config.json')
SOURCES = (PARENT, Path('outputs/emotion-block-night/diffa/gsm8k/run_config.json'))
RECIPE = dict(MODELS['phi4'], dtype='bfloat16', do_sample=False, num_beams=1,
              use_cache=True, repetition_penalty=1., text_only_output=True)


def registry_root(explicit=None):
    if explicit is not None: return explicit
    for root in (Path('outputs/jenny-zip-audio'), Path('.')):
        if any((root/name).is_file() for name in ('metadata/audio_inventory.jsonl',
                                                 'outputs/shared-three-emotions/audio_inventory.jsonl')):
            return root
    raise FileNotFoundError('No existing GSM audio plan/inventory found. Supply --source CONFIG or --audio-root JENNY_DIRECTORY.')


def select(source=None, audio_root=None):
    source = source or next((p for p in SOURCES if p.exists()), None)
    if source is not None:
        parent = json.loads(source.read_text())
        if parent.get('task') != 'gsm8k' or len(parent['bank']) != TOTAL:
            raise ValueError('Use the complete1319 GSM night plan, not a held-out subset.')
        order, bank = parent['question_order'], parent['bank']
        if len(order) != TOTAL or len(set(order)) != TOTAL or set(order) != set(bank):
            raise ValueError('Incomplete/duplicate question order.')
        records = {}
        for r in parent['records']:
            key = r['question_id'], r.get('condition', r.get('style'))
            if key[1] not in STYLES: continue
            if key in records or key[0] not in bank or r['style'] != key[1]:
                raise ValueError('Duplicate/inconsistent source recordings.')
            records[key] = dict(r, condition=key[1])
        if set(records) != {(q,s) for q in order for s in STYLES}:
            raise ValueError('Source must contain neutral and angry for every question.')
        if audio_root is not None or any(not Path(r['audio_file']).is_file() for r in records.values()):
            registry = Registry(registry_root(audio_root))
            records = {k:dict(registry.relocate(r), condition=k[1]) for k,r in records.items()}
        receipt = dict(source_config=str(source.absolute()), source_config_sha256=sha(source))
    else:
        registry = Registry(registry_root(audio_root))
        bank, records = {}, {}
        for q in registry.neutral_ids('gsm8k'):
            neutral = registry.choose('gsm8k', q, 'neutral')
            bank[q] = dict(question_id=q, question=neutral['question'], answer=neutral['answer'])
            for s in STYLES:
                options = [x for x in registry.options['gsm8k',q,s]
                           if x['question'] == neutral['question'] and x['answer'] == neutral['answer']]
                if len({x['record']['audio_sha256'] for x in options}) != 1:
                    raise ValueError(f'Missing or ambiguous {q}/{s}; restore the same Jenny audio.')
                records[q,s] = dict(options[0]['record'], condition=s)
        if len(bank) != TOTAL: raise ValueError(f'Expected all1319 GSM questions; found {len(bank)}.')
        order = sorted(bank); random.Random(42).shuffle(order)
        receipt = dict(audio_root=str(registry.root))
    selected = [records[q,s] for q in order for s in STYLES]
    for r in selected:
        if (r['reference_answer'] != str(bank[r['question_id']]['answer']).rsplit('####',1)[-1].strip()
                or r['voice'] != 'en-US-JennyNeural' or sha(r['audio_file']) != r['audio_sha256']):
            raise ValueError('Source audio/voice/reference differs from the frozen bank.')
    return bank, order, selected, receipt


def prepare(output, source=None, audio_root=None):
    output.mkdir(parents=True, exist_ok=True)
    config = output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text()); validate_plan(plan)
        if source is not None and plan.get('source_config') != str(source.absolute()):
            raise ValueError('Source changed; use the original command or another output folder.')
        return plan
    bank, order, recordings, receipt = select(source, audio_root)
    plan = dict(protocol=PROTOCOL, task='gsm8k', model='phi', recipe=RECIPE, prompt=PROMPT,
        seed=42, max_new_tokens=512, question_order=order, bank=bank, recordings=recordings,
        styles=list(STYLES), interventions=[], strength=0., **receipt,
        selection='All1319 existing Jenny GSM questions, neutral and angry; no reserved/fitting questions, '
            'emotion-classifier, outcome or duration filtering. No TTS, activation capture or steering.',
        scoring='Installed upstream GSM8K strict/flexible filters and exact-match on saved answers; '
            'numeric equivalence and format compliance additional. Custom bare audio prompt, not full text-harness prompting.')
    winners.freeze(config, plan); winners.atomic_text(output/'prompt.txt', PROMPT+'\n')
    return plan


def validate_plan(plan):
    if (plan.get('protocol') != PROTOCOL or plan.get('prompt') != PROMPT or plan.get('recipe') != RECIPE
            or plan.get('seed') != 42 or plan.get('max_new_tokens') != 512 or plan.get('strength') != 0.
            or plan.get('interventions') != []):
        raise ValueError('Frozen Phi bare-evaluation settings differ.')


def answers(output, plan, repair=False):
    validate_plan(plan)
    expected = {(r['question_id'],r['condition']):r for r in plan['recordings']}
    checksum = sha(output/'run_config.json'); done = {}
    for r in read_rows(output/'results.jsonl', repair=repair):
        key = r['question_id'],r['condition']; record = expected.get(key)
        if (key in done or record is None or r.get('run_config_sha256') != checksum
                or any(r.get(k) != plan[k] for k in ('prompt','seed','max_new_tokens','interventions','strength'))
                or r.get('model') != RECIPE['model'] or r.get('revision') != RECIPE['revision']
                or any(r.get(k) != record[k] for k in ('style','voice','audio_sha256','reference_answer'))
                or type(r.get('response_tokens')) is not int or r['response_tokens'] < 0):
            raise ValueError('Duplicate/incompatible saved bare-prompt answer.')
        done[key] = r
    return done


def reuse_night(output, plan, source=PARENT):
    """Import only exact matching unsteered Phi bare answers; source files remain untouched."""
    if not source.exists(): return 0
    parent = json.loads(source.read_text())
    if (parent.get('model') != 'phi' or parent.get('prompt') != PROMPT or parent.get('seed') != 42
            or parent.get('max_new_tokens') != 512 or any(parent.get('recipe',{}).get(k) != RECIPE[k]
                for k in ('model','revision','dtype','do_sample','num_beams','use_cache','repetition_penalty'))):
        print('Existing night answers use different settings; generating this evaluation independently.', flush=True)
        return 0
    wanted = {(r['question_id'],r['condition']):r for r in plan['recordings']}
    done = answers(output, plan, repair=True); checksum = sha(output/'run_config.json')
    parent_hash = sha(source); imported = 0; seen = set()
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(source.parent/'deliveries.jsonl'):
            key = row['question_id'],row['condition']
            if key[1] not in STYLES: continue
            record = wanted.get(key)
            if (key in seen or record is None or row.get('run_config_sha256') != parent_hash
                    or row.get('model') != 'phi' or row.get('prompt') != PROMPT or row.get('seed') != 42
                    or row.get('max_new_tokens') != 512 or row.get('strength') != 0. or row.get('interventions') != []
                    or type(row.get('response_tokens')) is not int or row['response_tokens'] < 0
                    or any(row.get(k) != record[k] for k in ('style','voice','audio_sha256','reference_answer'))):
                raise ValueError('Existing Phi night response differs from its source/this recording.')
            seen.add(key)
            if key in done:
                if done[key]['response'] != row['response']: raise ValueError('Conflicting matching Phi bare response.')
                continue
            saved = dict(record, response=row['response'], response_tokens=row['response_tokens'],
                prompt=PROMPT, seed=42, max_new_tokens=512, model=RECIPE['model'], revision=RECIPE['revision'],
                interventions=[], strength=0., run_config_sha256=checksum,
                reused_from=str(source.parent/'deliveries.jsonl'), source_config_sha256=parent_hash)
            stream.write(json.dumps(saved)+'\n'); stream.flush(); imported += 1
    return imported


def generate(output, plan, native):
    import phi4_all_layer_diagnostic as hooks
    done = answers(output, plan, repair=True); checksum = sha(output/'run_config.json')
    with (output/'results.jsonl').open('a') as stream:
        for record in plan['recordings']:
            key = record['question_id'],record['condition']
            if key in done: continue
            print(f'Phi bare [{len(done)+1}/{len(plan["recordings"])}] {key[0]}/{key[1]}', flush=True)
            trace = hooks.generate(native, record, PROMPT, collect=False, interventions=[], max_new_tokens=512)
            row = dict(record, **trace, prompt=PROMPT, seed=42, max_new_tokens=512,
                model=RECIPE['model'], revision=RECIPE['revision'], strength=0., interventions=[],
                run_config_sha256=checksum, reused_from=None)
            stream.write(json.dumps(row)+'\n'); stream.flush(); done[key] = row
            print(f'  Saved tokens={row["response_tokens"]}; elapsed={row["generation_seconds"]:.1f}s', flush=True)


def report(output, plan):
    from rescore_audio_steering import paired
    raw = list(answers(output, plan).values())
    scored, provenance = winners.score_rows(dict(target=dict(task='gsm8k', bank=plan['bank'])), raw, output/'scoring')
    groups = {s:{r['question_id']:r for r in scored if r['condition'] == s} for s in STYLES}
    ids = sorted(groups['neutral'].keys() & groups['angry'].keys())
    print(f'\nPhi GSM8K bare: {len(raw)}/{len(plan["recordings"])} answers; {len(ids)}/{len(plan["question_order"])} matched neutral/angry pairs.', flush=True)
    lengths, comparisons = {}, []
    for s in STYLES:
        if not ids: continue
        tokens = [groups[s][q]['response_tokens'] for q in ids]
        lengths[s] = dict(n=len(ids), mean_tokens=mean(tokens), median_tokens=median(tokens))
        print(f'{s}: SAME N={len(ids)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}', flush=True)
        for m in groups[s][ids[0]]['metrics']:
            count = sum(groups[s][q]['metrics'][m] for q in ids)
            print(f'  {m}: {count}/{len(ids)} ({count/len(ids):.1%})', flush=True)
    if ids:
        for m in groups['neutral'][ids[0]]['metrics']:
            stats = paired([(groups['neutral'][q]['metrics'][m],groups['angry'][q]['metrics'][m]) for q in ids])
            comparisons.append(dict(metric=m, **stats))
            print(f'angry vs neutral / {m}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}', flush=True)
    winners.atomic_text(output/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'summary.json', json.dumps(dict(protocol=PROTOCOL, prompt=PROMPT,
        matched_questions=len(ids), expected_questions=len(plan['question_order']), lengths=lengths,
        paired=comparisons, scorer=provenance, note='Evaluation only; all questions; no fitting or steering. '
            'Matched completed pairs; exact paired McNemar p and paired bootstrap95% CI, unadjusted.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','run','summary'), default='run')
    parser.add_argument('--source', type=Path, help='Existing complete GSM night run_config.json (optional).')
    parser.add_argument('--audio-root', type=Path, help='Existing extracted Jenny ZIP; no new audio is synthesized.')
    parser.add_argument('--output', type=Path, default=Path('outputs/phi4-gsm-bare-eval'))
    args = parser.parse_args()
    config = args.output/'run_config.json'
    if args.mode == 'summary':
        if config.exists(): report(args.output, json.loads(config.read_text()))
        else: print('Phi bare GSM evaluation not prepared yet.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try: fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Phi bare evaluation already active; use --mode summary.')
        plan = prepare(args.output, args.source, args.audio_root)
        print(f'Phi bare GSM: {len(plan["question_order"])} questions × neutral/angry. Evaluation only.\n'+PROMPT, flush=True)
        if args.mode == 'prepare':
            winners.score_rows(dict(target=dict(task='gsm8k',bank=plan['bank'])), [], args.output/'scoring-preflight')
            print('Prepared; no generation, collection, classifier selection or steering.', flush=True); return
        print('Reused matching prior Phi bare answers:', reuse_night(args.output, plan), flush=True)
        done = answers(args.output, plan, repair=True)
        if len(done) < len(plan['recordings']):
            for r in plan['recordings']:
                if sha(r['audio_file']) != r['audio_sha256']: raise ValueError('Frozen audio missing/changed.')
            Path('outputs').mkdir(exist_ok=True)
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU.', flush=True); fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
                    time.sleep(30)
                from steer_phi4_gsm_reverse import NativePhi
                native = NativePhi(11)
                winners.freeze(args.output/'runtime.json', dict(recipe=RECIPE,
                    model_config=native.model.config.to_dict(), runner_sha256=sha(__file__),
                    packages={k:version(k) for k in ('torch','transformers')}))
                generate(args.output, plan, native)
        print('Phi generations saved. Use --mode summary with the scoring environment.', flush=True)


if __name__ == '__main__': main()
