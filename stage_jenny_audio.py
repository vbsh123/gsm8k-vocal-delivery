"""Stage the Jenny ZIP separately on Phi; leave all historical audio/configs unchanged."""
import argparse
from pathlib import Path

from restore_diffa_bundle import restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', nargs='?', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/jenny-zip-audio'))
    args = parser.parse_args()
    if args.archive is None:
        matches = sorted(Path('/workspace').glob('jenny-six-datasets-audio-*.zip'))
        if len(matches) != 1:
            raise ValueError('Expected one completed audio ZIP in /workspace; otherwise pass its exact path.')
        args.archive = matches[0]
    args.output.mkdir(parents=True, exist_ok=True)
    restore(args.archive, args.output)
    print('Staged Jenny audio at', args.output.resolve())


if __name__ == '__main__':
    main()
