"""CPU environment/disk check and pinned Step text-path download; no GPU load."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from bbh_candidate_adapters import MODELS,checkpoint


def check_environment():
    import torch,torchaudio,librosa
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen2Model
    for package in ('torch','torchaudio','transformers'):
        actual=version(package).split('+')[0]
        if actual!=MODELS['step'][package]:
            raise RuntimeError(f'Step expects {package}=={MODELS["step"][package]}; found {actual}. No existing environment modified.')
    print('Step text-path imports ready in existing DIFFA environment; no package installation.',flush=True)


def download(output):
    check_environment()
    path=Path(checkpoint('step'))
    files={str(p.relative_to(path)):p.stat().st_size for p in path.rglob('*') if p.is_file()}
    if any(name.startswith('token2wav/') for name in files):
        print('Existing optional token2wav files present; not loaded by this run.',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'checkpoint.json').write_text(json.dumps(dict(model=MODELS['step'],path=str(path),files=files,
        environment={p:version(p) for p in ('torch','torchaudio','transformers','librosa')},
        policy='Pinned text/audio understanding files only; no speech-synthesis download or model load.'),indent=2)+'\n')
    print(f'Step checkpoint ready: {path}; {sum(files.values())/2**30:.2f} GiB.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/step-gsm400'))
    args=parser.parse_args();download(args.output)
