"""CPU-only BBH saved-answer scoring with pinned EleutherAI harness code/config.

Executes upstream filter/metric bodies unchanged, extracting their AST nodes to
avoid installing the full harness/model stack. Only registry decorators and the
abstract Filter base are removed. This is audio-adapted scoring, not the official
text/few-shot generation protocol. Original response strings are never rewritten.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import unicodedata

import numpy as np
import yaml

from rescore_audio_steering import HARNESS, download, paired
from steer import read_rows

TASK = 'bbh_cot_zeroshot_logical_deduction_three_objects'
BASE = 'lm_eval/tasks/bbh/cot_zeroshot/'
FILES = {'task.yaml': BASE+'logical_deduction_three_objects.yaml',
         'template.yaml': BASE+'_cot_zeroshot_template_yaml',
         'utils.py': BASE+'utils.py', 'extraction.py': 'lm_eval/filters/extraction.py',
         'selection.py': 'lm_eval/filters/selection.py', 'metrics.py': 'lm_eval/api/metrics.py',
         'LICENSE.md': 'LICENSE.md'}
STYLES = ('neutral', 'angry', 'sad', 'happy')


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = []
    for name in names:
        node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        node.decorator_list = []
        if isinstance(node, ast.ClassDef) and name in ('RegexFilter', 'TakeFirstFilter'):
            node.bases = []
        selected.append(node)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future, *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


class HarnessScorer:
    def __init__(self, cache):
        cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
        revision_file = cache/'revision.txt'
        if not revision_file.exists():
            revision = json.loads(download('https://api.github.com/repos/EleutherAI/lm-evaluation-harness/commits/main'))['sha']
            revision_file.write_text(revision+'\n')
        revision = revision_file.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Invalid pinned harness revision')
        for name, remote in FILES.items():
            if not (cache/name).exists():
                (cache/name).write_bytes(download(HARNESS+revision+'/'+remote))
        self.provenance = dict(repository='EleutherAI/lm-evaluation-harness', revision=revision,
            task=TASK, files={name: hashlib.sha256((cache/name).read_bytes()).hexdigest() for name in FILES},
            adaptation='Unchanged upstream AST filter/metric bodies; registry decorators and abstract Filter base omitted. Raw responses and official parenthesized targets unchanged. No generation stop truncation added retrospectively.')
        provenance_file = cache/'provenance.json'
        if provenance_file.exists() and json.loads(provenance_file.read_text()) != self.provenance:
            raise ValueError('Cached upstream source changed; use a new scorer cache')
        provenance_file.write_text(json.dumps(self.provenance, indent=2)+'\n')
        class Loader(yaml.SafeLoader):
            pass
        Loader.add_constructor('!function', lambda loader,node: loader.construct_scalar(node))
        config = yaml.load((cache/'template.yaml').read_text(), Loader=Loader)
        config.update(yaml.load((cache/'task.yaml').read_text(), Loader=Loader))
        if config['task'] != TASK or config['num_fewshot'] != 0:
            raise ValueError('Unexpected upstream BBH task variant')
        metric, = config['metric_list']
        if metric['metric'] != 'exact_match':
            raise ValueError('Upstream metric changed')
        self.metric_options = {k:v for k,v in metric.items() if k not in ('metric','aggregation','higher_is_better')}
        namespace = dict(re=re, sys=sys, unicodedata=unicodedata, np=np, string=string)
        load_nodes(cache/'extraction.py', ['RegexFilter'], namespace)
        load_nodes(cache/'selection.py', ['TakeFirstFilter'], namespace)
        load_nodes(cache/'utils.py', ['ExtendedRegexFilter','MultiChoiceRegexFilter'], namespace)
        load_nodes(cache/'metrics.py', ['exact_match_hf_evaluate'], namespace)
        factories = {'regex': namespace['RegexFilter'], 'take_first': namespace['TakeFirstFilter'],
                     'utils.MultiChoiceRegexFilter': namespace['MultiChoiceRegexFilter']}
        self.pipelines = {}
        for entry in config['filter_list']:
            self.pipelines[entry['name']] = [factories[step['function']](**{k:v for k,v in step.items() if k != 'function'})
                                             for step in entry['filter']]
        self.metric = namespace['exact_match_hf_evaluate']

    def score(self, responses, docs):
        results = {}
        if len(responses) != len(docs):
            raise ValueError('Response/document count differs')
        for name, pipeline in self.pipelines.items():
            values = [[response] for response in responses]
            for filter_ in pipeline:
                values = list(filter_.apply(values, docs))
            results[name] = [dict(prediction=prediction,
                                 correct=bool(self.metric(predictions=[prediction], references=[doc['target']],
                                                          **self.metric_options)['exact_match']))
                             for prediction, doc in zip(values, docs, strict=True)]
        return results


def report_saved(output, source, destination=None, cache=None):
    from run_bbh_logical200 import SOURCE_SHA256
    output, source = Path(output), Path(source)
    destination = Path(destination) if destination else output/'harness'
    original = (source/'logical_deduction_three_objects-source.json').read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official BBH source checksum differs')
    official = json.loads(original)['examples']
    bank = {q['question_id']: q for q in read_rows(source/'questions.jsonl')}
    for q in bank.values():
        expected = official[q['source_index']]
        if (q['original_input'] != expected['input'] or q['original_target'] != expected['target']
                or q['answer'] != expected['target'][1]):
            raise ValueError('Selected question differs from official BBH source')
    latest = {}
    for row in read_rows(output/'results.jsonl'):
        condition = row['condition']
        if condition not in STYLES:
            continue
        key = row['question_id'], row['voice'], condition
        if row['reference_answer'] != bank[key[0]]['answer']:
            raise ValueError('Saved reference differs from original BBH target')
        if key in latest and latest[key]['response'] != row['response']:
            raise ValueError('Conflicting saved responses for the same question/voice/style')
        latest[key] = row
    groups = {style:{k[:2] for k in latest if k[2] == style} for style in STYLES}
    matched = sorted(set.intersection(*groups.values()))
    keys = [(*key, style) for key in matched for style in STYLES]
    rows = [latest[key] for key in keys]
    destination.mkdir(parents=True, exist_ok=True)
    scorer = HarnessScorer(cache or destination/'upstream')
    docs = [dict(input=bank[r['question_id']]['original_input'],
                 target=bank[r['question_id']]['original_target']) for r in rows]
    scores = scorer.score([r['response'] for r in rows], docs)
    rescored = [dict(row, harness={name: values[i] for name,values in scores.items()}) for i,row in enumerate(rows)]
    accuracy, comparisons = [], []
    print('\nBBH: upstream EleutherAI harness CoT zero-shot filters + exact_match metric.')
    print('Audio-adapted saved generations; not a full official text evaluation. Responses are scored verbatim.')
    print(f'{len(matched)} questions/voices completed in ALL FOUR deliveries; no outcome/classifier filtering.')
    for metric in scores:
        by_style = {style:{(r['question_id'],r['voice']): r['harness'][metric]['correct']
                           for r in rescored if r['condition'] == style} for style in STYLES}
        for style,values in by_style.items():
            n = len(values); correct = sum(values.values())
            print(f'{metric:18s} {style:8s} {correct}/{n}' + (f' ({correct/n:.1%})' if n else ''))
            accuracy.append(dict(metric=metric, condition=style, n=n, correct=correct))
        if matched:
            for style in STYLES[1:]:
                result = paired([(by_style['neutral'][k], by_style[style][k]) for k in matched])
                comparisons.append(dict(metric=metric, condition=style, **result))
                print(f'  {style} vs neutral: {result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}')
    summary = dict(scorer=scorer.provenance, matched_questions=len(matched),
                   unmatched_rows_excluded=len(latest)-len(rows), accuracy=accuracy, paired=comparisons,
                   inference_protocol='Existing audio prompts/native DIFFA decoding unchanged; evaluator migration only')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (destination/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rescored))
    print('Strict extraction expects the upstream answer phrase; the existing Final answer prompt may not satisfy it.')
    print('Flexible extraction may match choices mentioned in reasoning; upstream behavior is preserved. Paired tests are unadjusted.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    parser.add_argument('--output', type=Path, default=Path('outputs/bbh-logical200-diffa2'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--upstream-cache', type=Path)
    args = parser.parse_args()
    report_saved(args.output, args.source, args.destination, args.upstream_cache)


if __name__ == '__main__':
    main()
