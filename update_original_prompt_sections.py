"""Add exact original prompts and original-prompt happy results to the offline report.

Reads archived configs and a separately identified result reference. Does not score,
generate, change source evidence, or replace existing benchmark tables.
"""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BEGIN = '<!-- original-prompts-and-happy:start -->'
END = '<!-- original-prompts-and-happy:end -->'
LABELS = dict(gsm8k='GSM8K', svamp='SVAMP', multiarith='MultiArith',
              bbh3='BBH · three objects', bbh5='BBH · five objects', ifeval='IFEval')


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(
        '<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows
    )+'</tbody></table></div>'


def section(root, report, reference):
    blocks = [BEGIN, '<section id="original-prompts"><h2>Original experiment prompts</h2>',
        '<p>Exact generation instructions from the archived original runs. These are custom audio '
        'adaptations; the saved-answer evaluator does not make the full generation prompt official. '
        'The question and choices, where applicable, are supplied in the audio.</p>']
    receipts = []
    for task, run in report['core'].items():
        config = root/run['config_file']
        saved = json.loads(config.read_text())
        prompt = saved.get('prompt', saved.get('wrapper'))
        if not isinstance(prompt, str) or prompt != run['prompt']:
            raise ValueError(f'Report/config prompt mismatch: {task}')
        collection = saved.get('collection_prompt')
        vector_path = None
        if collection is None:
            # Read the actual saved vector metadata rather than infer from a runner constant.
            import torch
            vector_path = root/next(v['file'] for v in report['vectors'] if v['task'] == task)
            collection = torch.load(vector_path, map_location='cpu', weights_only=True)['prompt']
        if not isinstance(collection, str):
            raise ValueError(f'Missing saved collection prompt: {task}')
        rel = str(config.relative_to(root))
        receipts.append(dict(task=task, evaluation_prompt=prompt, collection_prompt=collection,
            config_file=rel, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            collection_vector_file=str(vector_path.relative_to(root)) if vector_path else None))
        blocks.extend(['<details open><summary><strong>'+escape(LABELS[task])+'</strong></summary>',
            '<h3>Evaluation · neutral, emotional audio and steered neutral</h3><pre>'+escape(prompt)+'</pre>'])
        if collection == prompt:
            blocks.append('<p>Vector collection used the same prompt.</p>')
        else:
            blocks.append('<h3>Vector collection · neutral and emotional audio</h3><pre>'+escape(collection)+'</pre>')
        blocks.append('<p><a href="../../'+escape(rel, quote=True)+'">Archived run configuration</a></p></details>')
    blocks.extend(['</section>', '<section id="original-happy"><h2>Happy · original prompt</h2>',
        '<p>Happy versus its matched neutral baseline under the original prompts above. '
        'These are the original happy runs, with their original fitting and evaluation cohorts. '
        'GSM8K and SVAMP delivery summaries are also present in the downloaded archive; '
        'the remaining entries are preserved user-pasted console summaries. '
        'They are not pooled with the separate angry tables or with later prompt experiments. '
        'P-values are unadjusted paired tests; confidence intervals are the reported 95% intervals.</p>'])
    for mode, title in [('delivery', 'Natural happy audio'), ('steering', 'Neutral audio + happy vector · L31 ×1')]:
        blocks.append('<h3>'+title+'</h3>')
        result_rows, lengths = [], []
        for run in reference['runs']:
            if run['mode'] != mode:
                continue
            scope = 'Collection only · partial' if run['scope'] == 'collection' else 'Held-out'
            for row in run['metrics']:
                n = run['n']
                b, h = row['neutral_correct'], row['happy_correct']
                lo, hi = row['ci95_gain_pp']
                result_rows.append([LABELS[run['task']], scope, row['metric'], n,
                    f'{b}/{n} ({100*b/n:.1f}%)', f'{h}/{n} ({100*h/n:.1f}%)',
                    f'{100*(h-b)/n:+.2f}', f'[{lo:+.2f}, {hi:+.2f}]', f'{row["p"]:.4g}'])
            lengths.append([LABELS[run['task']], scope, run['n'],
                f'{run["neutral_tokens"][0]:.1f} / {run["neutral_tokens"][1]:g}',
                f'{run["happy_tokens"][0]:.1f} / {run["happy_tokens"][1]:g}'])
        blocks.append(table(['Task', 'Cohort', 'Metric', 'N', 'Matched neutral', 'Happy' if mode == 'delivery' else 'Steered',
                             'Gain pp', '95% CI pp', 'p'], result_rows))
        blocks.append(table(['Task', 'Cohort', 'N', 'Matched neutral tokens · mean / median',
                             ('Happy' if mode == 'delivery' else 'Steered')+' tokens · mean / median'], lengths))
    blocks.append('<p>IFEval: only 226/254 collection pairs were available in the pasted snapshot; '
        'no held-out happy delivery or happy steering result was reported. Original-prompt sad results '
        'are unavailable here. These gaps are not assigned zero scores.</p>'
        '<p><a href="data/original-prompt-happy-reference.json">Happy result reference and provenance</a> · '
        '<a href="data/original-generation-prompts.json">Exact prompt text and configuration hashes</a></p></section>'+END)
    return ''.join(blocks), receipts


def update(root, reference_path=HERE/'original_prompt_happy_reference.json'):
    folder = root/'07-reports/production'
    report = json.loads((folder/'data/report-data.json').read_text())
    reference = json.loads(reference_path.read_text())
    content, receipts = section(root, report, reference)
    paths = [root/'07-reports/runs.html', folder/'index.html']
    rendered = []
    for path in paths:
        html = path.read_text()
        html = re.sub(re.escape(BEGIN)+'.*?'+re.escape(END), '', html, flags=re.S)
        if html.count('<section id="protocols">') != 1:
            raise ValueError(f'Expected one protocol section: {path}')
        html = html.replace('<section id="protocols">', content+'<section id="protocols">', 1)
        if '<a href="#original-prompts">' not in html:
            html = html.replace('<a href="#protocols">', '<a href="#original-prompts">Original prompts</a>'
                '<a href="#original-happy">Original happy runs</a><a href="#protocols">', 1)
        rendered.append((path, html))
    for path, html in rendered:
        path.write_text(html)
    (folder/'data/original-prompt-happy-reference.json').write_text(json.dumps(reference, indent=2)+'\n')
    (folder/'data/original-generation-prompts.json').write_text(json.dumps(receipts, indent=2)+'\n')
    print('Updated original prompts and original-prompt happy results:', paths[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/c/Users/User/Downloads/emotional-audio'))
    args = parser.parse_args()
    update(args.root)
