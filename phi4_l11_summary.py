"""Compact CPU-only flexible and strict summary for Phi L11 answer x-1."""
import argparse
from pathlib import Path
import re
import subprocess


def extract(output, metric='flexible-extract'):
    prefix = re.escape(metric)
    neutral = re.search(r'^' + prefix + r' / neutral: (.+)$', output, re.M)
    steered = re.search(r'^' + prefix + r' / neutral-[^\n]+: (.+)\n\s*gain=([^;\n]+); p=([^;\n]+); CI=([^\n]+)', output, re.M)
    if not steered:
        status = next((line for line in output.splitlines() if 'not run/prepared' in line), None)
        completed = re.findall(r'^\d+ questions completed.*$', output, re.M)
        return status or (completed[-1] if completed else f'No {metric} result saved yet.')
    return (f'neutral: {neutral.group(1) if neutral else "unknown"}\n'
            f'  L11: {steered.group(1)} | gain={steered.group(2)} pp | '
            f'p={steered.group(3)} | CI={steered.group(4)}')


def main(default_strengths=None, default_datasets=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=default_datasets or ['gsm8k', 'multiarith', 'svamp', 'bbh3', 'bbh5'])
    parser.add_argument('--strengths', type=float, nargs='+', default=default_strengths or [-1])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    log = root / 'outputs/phi4-l11-summary.stderr.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with log.open('w') as errors:
        for dataset, strength in ((d, s) for d in args.datasets for s in args.strengths):
            print(f'\n{dataset} — L11 answer x{strength:g}', flush=True)
            result = subprocess.run(['bash', 'run_phi4_sweep.sh', '--mode', 'summary',
                                     '--datasets', dataset, '--layers', '11', '--group', 'answer',
                                     '--strengths', f'{strength:g}'], cwd=root, text=True,
                                    stdout=subprocess.PIPE, stderr=errors)
            if result.returncode:
                failed = True
                print(f'Summary failed (exit {result.returncode}); see {log}.')
                print('\n'.join(result.stdout.splitlines()[-4:]))
            else:
                if dataset == 'ifeval':
                    for line in result.stdout.splitlines():
                        if (line.startswith(('Official IFEval checks:', 'Condition', 'neutral '))
                                or re.match(r'^neutral-ifeval-.*(?: /(?:strict|loose)_prompt:|\s+\d+)', line)
                                or 'not run/prepared' in line):
                            print(line)
                    continue
                for metric in ('flexible-extract', 'strict-match'):
                    print(f'  {metric}')
                    print(extract(result.stdout, metric))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
