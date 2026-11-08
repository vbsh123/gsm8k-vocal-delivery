"""One offline HTML: separate steering charts plus matched delivery comparisons.

Reads saved answers only. Supports source folders, metadata archives, and loose
Qwen/Kimi downloads whose browser-added filenames have lost their parent folders.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import json
from math import comb
from pathlib import Path
import tarfile
import zipfile

from plot_azure_steering import pasted_data, six_question_data, from_runs
from scoring import score_response, SCORER_VERSION

MODELS = {'DIFFA-2': 'results-matched-accuracy.jsonl', 'Qwen': 'results-qwen-omni.jsonl',
          'Kimi': 'results-kimi-audio.jsonl'}
CLASSIFIERS = ('emotion2vec', 'sensevoice')
FILTERS = [('unfiltered', 'Unfiltered'), ('emotion2vec', 'emotion2vec agreed'),
           ('both', 'Both classifiers agree'), ('either', 'At least one classifier agrees')]
STYLES = ('neutral', 'angry', 'happy', 'sad')
METADATA = {'plan.jsonl', *MODELS.values(), *(f'results-{c}.jsonl' for c in CLASSIFIERS)}


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_metadata(paths, archives):
    files = {}
    def add(name, content):
        path = Path(name)
        if path.name not in METADATA or '.ipynb_checkpoints' in path.parts:
            return
        key = (path.parent.name, path.name)
        if key in files and files[key] != content:
            raise ValueError(f'Conflicting metadata sources: {key}')
        files[key] = content
    for folder in paths:
        for name in METADATA:
            path = folder / name
            if path.exists():
                add(str(path), path.read_text())
    for path in archives:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f'Invalid archive CRC: {path}')
                for name in archive.namelist():
                    if Path(name).name in METADATA:
                        add(name, archive.read(name).decode())
        else:
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile() and Path(member.name).name in METADATA:
                        add(member.name, archive.extractfile(member).read().decode())
    return files


def agrees(style, labels, rule):
    votes = [labels.get(name) == style for name in CLASSIFIERS]
    return {'unfiltered': True, 'emotion2vec': votes[0],
            'both': all(votes), 'either': any(votes)}[rule]


def paired_comparisons(tasks, answers, labels):
    """Intersect question+voice keys separately for every style pair/filter."""
    tables = []
    for rule, title in FILTERS:
        groups = {style: {} for style in STYLES}
        for key, task in tasks.items():
            if key not in answers or task['style'] not in groups:
                continue
            if agrees(task['style'], labels.get(key, {}), rule):
                row = answers[key]
                groups[task['style']][task['question_id'], task['voice']] = row
        for reference, test in combinations(STYLES, 2):
            left, right = groups[reference], groups[test]
            keys = sorted(left.keys() & right.keys())
            wins = losses = ref_correct = test_correct = 0
            for key in keys:
                a, b = left[key], right[key]
                if Fraction(a['reference_answer']) != Fraction(b['reference_answer']):
                    raise ValueError(f'Different reference answers for matched question {key}')
                ref_correct += a['ok']
                test_correct += b['ok']
                wins += b['ok'] and not a['ok']
                losses += a['ok'] and not b['ok']
            n, discordant = len(keys), wins + losses
            p = min(1., 2*sum(comb(discordant, k) for k in range(min(wins, losses)+1))/2**discordant) if discordant else 1.
            tables.append(dict(filter=rule, filter_label=title, reference=reference, test=test, n=n,
                               ref_correct=ref_correct, test_correct=test_correct, wins=wins, losses=losses,
                               p=p if n else None, pair_ids=[list(key) for key in keys]))
    return tables


def delivery_data(files, loose_answers):
    tasks, folder_audio, labels = {}, {}, defaultdict(dict)
    for (folder, name), content in files.items():
        if name != 'plan.jsonl':
            continue
        for row in jsonl(content):
            if row.get('provider') != 'azure' or row['style'] not in STYLES:
                continue
            key = (row['question_id'], row['voice'], row['style'])
            # A question/style/voice may not silently belong to two batches.
            if key in tasks:
                raise ValueError(f'Duplicate planned recording: {key}')
            batch = 'Batch 1 · questions 50–549' if folder in {'azure-jenny500-extra', 'azure-jenny500-neutral'} else folder
            tasks[key] = {**row, 'batch': batch, 'folder': folder}
            folder_audio[folder, row['audio_path']] = key
    if not tasks:
        raise ValueError('No Azure plan.jsonl files found')
    for (folder, name), content in files.items():
        for classifier in CLASSIFIERS:
            if name != f'results-{classifier}.jsonl':
                continue
            for row in jsonl(content):
                key = folder_audio.get((folder, row['audio_path']))
                if key is not None:
                    previous = labels[key].get(classifier)
                    if previous is not None and previous != row['emotion']:
                        raise ValueError(f'Conflicting classifier labels: {key}')
                    labels[key][classifier] = row['emotion']
    results = {model: {} for model in MODELS}
    provenance = {model: set() for model in MODELS}
    def add(model, row, key, source):
        if key is None or key not in tasks:
            raise ValueError(f'Answer has no unique matching Azure plan: {source}')
        task = tasks[key]
        if (row.get('question_id') != task['question_id'] or row.get('style') != task['style']
                or row.get('voice') != task['voice']):
            raise ValueError(f'Answer metadata disagrees with plan: {key}')
        gold = str(row['reference_answer']).replace(',', '').strip()
        planned = str(task['answer']).rsplit('####', 1)[-1].replace(',', '').strip()
        if Fraction(gold) != Fraction(planned):
            raise ValueError(f'Reference answer differs from plan: {key}')
        value = dict(ok=score_response(row['response'], gold)['correct'], reference_answer=gold,
                     response=row['response'], model=row.get('model'), revision=row.get('revision'),
                     prompt=row.get('prompt'), seed=row.get('seed'), max_new_tokens=row.get('max_new_tokens'))
        if key in results[model] and results[model][key] != value:
            raise ValueError(f'Conflicting saved answers/recipes: {model}: {key}')
        results[model][key] = value
        provenance[model].add(source)
    for (folder, name), content in files.items():
        for model, filename in MODELS.items():
            if name == filename:
                for row in jsonl(content):
                    key = folder_audio.get((folder, row['audio_path']))
                    if key is not None:
                        add(model, row, key, f'{folder}/{name}')
    for model, path in loose_answers:
        for row in jsonl(path.read_text()):
            key = (row['question_id'], row['voice'], row['style'])
            if key not in tasks or Path(row['audio_path']).name != Path(tasks[key]['audio_path']).name:
                raise ValueError(f'Loose answer does not match planned recording: {path}: {key}')
            add(model, row, key, path.name)
    batches = sorted({task['batch'] for task in tasks.values()})
    model_data = []
    for model in MODELS:
        cohorts = []
        for batch in batches:
            cohort = {key: task for key, task in tasks.items() if task['batch'] == batch}
            answer_count = sum(key in results[model] for key in cohort)
            if answer_count:
                cohorts.append(dict(name=batch, answers=answer_count,
                                    comparisons=paired_comparisons(cohort, results[model], labels)))
        model_data.append(dict(name=model, cohorts=cohorts, sources=sorted(provenance[model]),
                               answer_count=len(results[model])))
    return dict(models=model_data, scorer=SCORER_VERSION,
                classifiers=list(CLASSIFIERS), filters=[dict(id=k, label=v) for k, v in FILTERS])


def render_report(steering, deliveries):
    steering=list(steering)
    seen={e.get('id') for e in steering}
    steering.extend(e for e in followup_steering() if e['id'] not in seen)
    steering=[explain_experiment(e) for e in steering]
    template = Path(__file__).with_name('jenny_report.html').read_text()
    payload = json.dumps(dict(steering=steering, deliveries=deliveries, audio_speed=audio_speed_data()), ensure_ascii=True, allow_nan=False)
    return template.replace('__JENNY_REPORT_DATA__', payload.replace('<', '\\u003c'))


def explain_experiment(experiment):
    """Reader-facing descriptions; keep identifiers and numerical results intact."""
    descriptions={
        'initial-pilot-50':('Initial search: where to apply the anger direction?',
            'An anger direction was learned from 425 matched angry/neutral pairs in recording set A. We then steered the neutral audio for 50 questions from the separate recording set B, testing layers0–25 and audio/answer representations separately at strength1. Each condition is compared with the same neutral recording without steering.'),
        'batch1-neutral-487':('Transfer test: direction from 414 pairs → 487 different neutral recordings',
            'We learned a new anger direction from 414 matched angry/neutral pairs in set B, then applied it to 487 neutral recordings in set A. L11 answer and L12 acoustic steering were tested separately at strength1. The reported percentage is math accuracy, compared with the same recordings without steering. These questions were not used to extract this direction.'),
        'batch2-neutral-460':('Transfer test: direction from 425 pairs → 460 different neutral recordings',
            'We learned an anger direction from 425 matched angry/neutral pairs in set A, then applied it to 460 neutral recordings in set B. Each question had evaluated neutral and angry deliveries; steering was applied only to neutral audio. The percentage is math accuracy against the same recordings without steering. Layer choices had been explored on subsets of this test set.'),
        'recovery-74':('Can steering repair answers that angry delivery got right? — 74 questions',
            'Using the direction learned from set A, we tested 74 questions in set B for which neutral delivery was incorrect and angry delivery was correct. These exclude the initial50-question search. All74 were also incorrect under the fresh unsteered baseline. Here the percentage means the fraction of these failures repaired by steering; it is not overall math accuracy.'),
        'regressions-260':('Does steering damage answers that were already correct? — 260 questions',
            'Using the direction learned from set A, we tested 260 questions in set B for which both neutral and angry delivery had been correct. The fresh unsteered baseline was correct on all260. Here the percentage means previously correct answers retained after steering; a drop from100% is a regression. This deliberately selected success set is not an overall accuracy test.'),
        'six-question-combinations':('Small diagnostic: stronger, weaker and combined interventions — 6 questions',
            'Six questions from the initial search were selected because neutral delivery failed while angry delivery succeeded. We tested individual layers and combinations using the set A direction. These same questions guided the settings, so this panel is a diagnostic rather than an independent evaluation. For a combination with different strengths, x=1 means the full list of strengths shown in its name.'),
        'recovery74-promoted':('Expanded layer search: selected candidates on all74 recovery questions',
            'We first screened previously untested layers on ten of the74 recovery questions. Thirteen layer/group choices repaired at least4/10 and were tested at strength1.25 on all74. This panel includes those ten selection questions. Every question originally failed with neutral delivery and succeeded with angry delivery; the percentage is failures repaired, not overall accuracy.'),
        'recovery64-promoted':('Follow-up beyond the screen: selected candidates on the other64 questions',
            'These are the same thirteen candidates selected on ten questions, now reported only on the other64. The ten screening questions are excluded. The direction is learned from set A and applied to neutral recordings in set B at strength1.25. The questions are still selected for neutral failure and angry success; the percentage measures recovery on that subset.'),
        'recovery74-L16-L23-strengths':('How much steering? — layers16 and23 on the same74 failures',
            'We compared strengths0.5,1 and1.25 for answer-token steering at layers16 and23, using the set A direction on the same74 recovery questions in set B. The percentage is the fraction of unsteered failures repaired. All74 include the ten screening questions. Benefits on this failure set must be weighed against losses on previously correct answers.'),
    }
    entry=descriptions.get(experiment.get('id'))
    if entry is None:
        return experiment
    title,source=entry
    # Keep spaces around counts/coefficients readable in prose.
    for a,b in [('layers0','layers 0'),('strength1','strength 1'),('strengths0','strengths 0'),
                ('all74','all 74'),('all260','all 260'),('initial50','initial 50'),
                ('at least4','at least 4'),('other64','other 64'),('layers16','layers 16'),
                ('and23','and 23'),('from100','from 100'),('the74','the 74'),
                ('same74','same 74'),('0.5,1 and1.25','0.5, 1 and 1.25')]:
        title,source=title.replace(a,b),source.replace(a,b)
    return {**experiment,'title':title,'source':source}


def followup_steering():
    from plot_azure_steering import point
    completed=[(0,'acoustic',18,14),(1,'acoustic',18,14),(3,'acoustic',17,13),
               (5,'answer',18,13),(7,'acoustic',15,11),(9,'acoustic',18,14),
               (10,'acoustic',21,17),(11,'acoustic',22,18),(13,'answer',16,12),
               (16,'answer',25,21),(17,'answer',21,17),(23,'answer',25,21),(26,'answer',17,13)]
    layers={str(l):dict(n=74,baseline=0,points=[point(g,1.25,k,74,k,0)]) for l,g,k,_ in completed}
    remaining={str(l):dict(n=64,baseline=0,points=[point(g,1.25,k,64,k,0)]) for l,g,_,k in completed}
    strength={}
    for l,a,k in [(16,.5,14),(16,1,19),(16,1.25,25),(23,.5,16),(23,1,21),(23,1.25,25)]:
        strength.setdefault(str(l),dict(n=74,baseline=0,points=[]))['points'].append(point('answer',a,k,74,k,0))
    return [dict(id='recovery74-promoted',title='Promoted recovery layers ×1.25 · all74 · complete',layers=layers,
                 source='Completed pasted all74 table. Includes the ten screening questions used to select these targets. All74 are outcome-selected angry-correct / neutral-wrong questions; baseline0/74.'),
            dict(id='recovery64-promoted',title='Promoted recovery layers ×1.25 · remaining64 · complete',layers=remaining,
                 source='Completed pasted remaining64 table. Excludes the ten screening questions used to choose these conditions. Still selected for angry success / neutral failure; this is recovery, not overall model accuracy. Baseline0/64.'),
            dict(id='recovery74-L16-L23-strengths',title='L16/L23 answer · strength comparison · 74 questions',layers=strength,
                 source='Completed pasted results on the same74 outcome-selected neutral failures at ×0.5, ×1 and ×1.25. Includes the ten screening questions. Tiny paired p-values on this selected failure set do not establish general benefit.')]


def audio_speed_data():
    from plot_azure_steering import point
    sweep=[(.7,73,13,9),(.8,65,9,13),(.9,72,15,12),(1,69,0,0),
           (1.1,65,9,13),(1.2,69,11,11),(1.3,66,11,14),(1.4,62,7,14),(1.5,62,16,23)]
    points=[dict(rate=a,**point('speed',a,k,100,w,l)) for a,k,w,l in sweep]
    batches=[]
    for title,items in [('Recording set A — four requested deliveries', [('neutral',16.52,15.86),('angry',18.80,18.05),('happy',16.54,15.93),('sad',20.36,19.46)]),
                        ('Recording set B — six requested deliveries',[('neutral',16.97,15.98),('angry',19.33,18.39),('happy',17.00,16.07),('sad',20.94,19.74),('terrified',14.97,14.21),('shouting',19.81,18.82)])]:
        baseline=items[0][1]
        batches.append(dict(title=title,n=500,rows=[dict(style=s,mean_seconds=mean,median_seconds=median,
                            duration_change=100*(mean/baseline-1),relative_rate=baseline/mean) for s,mean,median in items]))
    return dict(n=100,baseline=69,points=points,durations=batches,
                source='DIFFA-2 answered the same 100 original Spoken GSM recordings at nine playback rates. Pitch is preserved, 1× uses unchanged audio, and seed 42 is reset before every answer. Only recordings up to 21 seconds long were selected so even the slowest rate, 0.7×, stays within the 30-second input limit without cropping.',
                original_steering=dict(n=300,baseline=210,point=point('acoustic',1,212,300,13,11)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, nargs='*', default=[])
    parser.add_argument('--archives', type=Path, nargs='*', default=[])
    parser.add_argument('--answers', action='append', default=[], metavar='MODEL=FILE',
                        help='Loose saved answers: Qwen=file.jsonl or Kimi=file.jsonl; repeat as needed')
    parser.add_argument('--steering-runs', type=Path, nargs='+')
    parser.add_argument('--update-report',type=Path,help='Reuse saved steering/model tables from an existing generated HTML, then add the latest experiment sections.')
    parser.add_argument('--output', type=Path, default=Path('outputs/azure-jenny-report.html'))
    args = parser.parse_args()
    if not args.paths and not args.archives:
        args.paths = [Path('outputs/azure-jenny500-extra'), Path('outputs/azure-jenny500-neutral'), Path('outputs/azure-jenny-next500')]
    loose = []
    for argument in args.answers:
        model, filename = argument.split('=', 1)
        if model not in MODELS:
            parser.error(f'Unknown model: {model}; choose from {list(MODELS)}')
        loose.append((model, Path(filename)))
    if args.update_report:
        text=args.update_report.read_text().split('const DATA=',1)[1]
        previous,_=json.JSONDecoder().raw_decode(text.lstrip())
        deliveries,steering=previous['deliveries'],previous['steering']
    else:
        deliveries = delivery_data(read_metadata(args.paths, args.archives), loose)
        steering = from_runs(args.steering_runs) if args.steering_runs else pasted_data() + [six_question_data()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(steering, deliveries))
    print(f'Saved {args.output}: steering panels, matched model tables, and audio-speed section')
    for model in deliveries['models']:
        print(f'{model["name"]}: {model["answer_count"]} saved answers, {len(model["cohorts"])} cohorts')


if __name__ == '__main__':
    main()
