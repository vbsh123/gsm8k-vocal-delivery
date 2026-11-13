"""DIFFA pitch controls on matched Azure neutral/angry recordings; no text changes."""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

from compare_cloud_accuracy import PROMPT
from measure_audio_pitch import load_audio, measure_pitch, PITCH_SETTINGS
from scoring import score_response
from steer import read_rows

CONDITIONS=['neutral','angry','neutral_to_angry_pitch','angry_to_neutral_pitch']


class PitchTransformError(ValueError):
    """A rendered recording failed the fixed audio-quality criteria."""


def select_pairs(paths,sf):
    groups={}
    for folder in paths:
        for row in read_rows(folder/'plan.jsonl'):
            if row.get('provider')!='azure' or row['style'] not in ('neutral','angry'):
                continue
            path=(folder/row['audio_path']).resolve()
            stat=path.stat();info=sf.info(path)
            if not 0<info.duration<=30:
                continue
            key=row['question_id'],row['voice']
            if row['style'] in groups.setdefault(key,{}):
                raise ValueError(f'Duplicate question/voice/style: {key}')
            groups[key][row['style']]=dict(task=row,audio_file=str(path),audio_bytes=stat.st_size,
                audio_mtime_ns=stat.st_mtime_ns,source_folder=str(folder.resolve()))
    pairs=[]
    for (qid,voice),clips in sorted(groups.items()):
        if set(clips)!=set(('neutral','angry')):
            continue
        if str(clips['neutral']['task']['answer'])!=str(clips['angry']['task']['answer']):
            raise ValueError(f'Reference answers differ: {qid}')
        pairs.append(dict(pair_id=qid+'__'+voice,question_id=qid,voice=voice,**clips))
    return pairs


def pitch_offset(neutral,angry):
    for value in (neutral,angry):
        if value['median_f0_hz'] is None or value['voiced_frames']<10 or value['voiced_fraction']<.1:
            return None
    delta=12*math.log2(angry['median_f0_hz']/neutral['median_f0_hz'])
    return delta if abs(delta)<=24 else None


def shift_audio(audio,semitones,path,ffmpeg,sf,np):
    ratio=2**(semitones/12)
    temporary=path.with_suffix('.tmp.wav')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y',
        '-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-af',
        f'rubberband=tempo=1:pitch={ratio:.12g}:formant=preserved:pitchq=quality',
        '-c:a','pcm_f32le',str(temporary)],input=np.asarray(audio,dtype='<f4').tobytes(),check=True)
    changed,sr=sf.read(temporary,dtype='float32')
    if sr!=16000 or abs(len(changed)-len(audio))>320 or len(changed)>30*16000:
        raise PitchTransformError(f'{path}: input={len(audio)/16000:.6f}s, output={len(changed)/sr:.6f}s, '
                                  f'sample_rate={sr}; requires 16 kHz, duration change <=20 ms and duration <=30s')
    original_rms=float(np.sqrt(np.mean(audio**2)))
    changed_rms=float(np.sqrt(np.mean(changed**2)))
    if not changed_rms:
        raise PitchTransformError(f'Pitch shift produced silent audio: {path}')
    changed*=original_rms/changed_rms
    sf.write(temporary,changed,16000,subtype='FLOAT')
    temporary.replace(path)
    return dict(duration_seconds=len(changed)/16000,rms=float(np.sqrt(np.mean(changed**2))),
                peak=float(np.max(np.abs(changed))),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(args):
    import imageio_ffmpeg
    import librosa
    import numpy as np
    import soundfile as sf
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    filters=subprocess.check_output([ffmpeg,'-hide_banner','-filters'],text=True,stderr=subprocess.DEVNULL)
    if 'rubberband' not in filters:
        raise ValueError('FFmpeg lacks Rubber Band; install imageio-ffmpeg==0.8.1 in this environment')
    pairs=select_pairs(args.paths,sf)
    if args.limit:
        pairs=pairs[:args.limit]
    if not pairs:
        raise ValueError('No matched, duration-eligible neutral/angry recordings')
    args.output.mkdir(parents=True,exist_ok=True)
    settings=dict(pairs=pairs,seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',
                  pitch_settings=PITCH_SETTINGS,librosa=librosa.__version__,
                  ffmpeg_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
                  filter='rubberband=tempo=1:pitch=2**(semitones/12):formant=preserved:pitchq=quality',
                  normalization='Preserve source RMS; float WAV; no clipping, padding or trimming',
                  minimum_voiced_frames=10,minimum_voiced_fraction=.1,maximum_absolute_shift_semitones=24,
                  duration_tolerance_seconds=.02)
    config=args.output/'run_config.json'
    if config.exists() and json.loads(config.read_text())!=settings:
        raise ValueError('Pitch experiment inputs or settings changed; use another output folder')
    config.write_text(json.dumps(settings,indent=2)+'\n')
    audio_folder=args.output/'audio';audio_folder.mkdir(exist_ok=True)
    cache={r['pair_id']:r for r in read_rows(args.output/'pairs.jsonl',repair=True)}
    for index,pair in enumerate(pairs,1):
        old=cache.get(pair['pair_id'])
        if old and (old['status']!='prepared' or all(Path(p).is_file() for p in old['shifted'].values())):
            continue
        print(f'Pitch [{index}/{len(pairs)}] {pair["question_id"]}: estimate original F0 and render controls on CPU.',flush=True)
        neutral=load_audio(pair['neutral']['audio_file'],sf,librosa)
        angry=load_audio(pair['angry']['audio_file'],sf,librosa)
        ns,_=measure_pitch(neutral,librosa,np);ans,_=measure_pitch(angry,librosa,np)
        delta=pitch_offset(ns,ans)
        record=dict(pair_id=pair['pair_id'],neutral_pitch=ns,angry_pitch=ans,shift_semitones=delta)
        if delta is None:
            record['status']='pitch_estimate_unusable'
        else:
            shifted={};metrics={};audit={}
            try:
                for name,audio,amount,target in [('neutral_to_angry_pitch',neutral,delta,ans),
                                               ('angry_to_neutral_pitch',angry,-delta,ns)]:
                    path=audio_folder/f'{pair["pair_id"]}__{name}.wav'
                    metrics[name]=shift_audio(audio,amount,path,ffmpeg,sf,np)
                    shifted[name]=str(path.resolve())
                    if index<=5:
                        stats,_=measure_pitch(load_audio(path,sf,librosa),librosa,np)
                        if stats['median_f0_hz'] is None:
                            raise ValueError('Pilot output has no measurable voiced pitch')
                        error=12*math.log2(stats['median_f0_hz']/target['median_f0_hz'])
                        audit[name]=dict(**stats,target_error_semitones=error)
                        if abs(error)>2:
                            raise ValueError(f'Pilot pitch missed its target by {error:.2f} semitones; inspect audio before proceeding')
            except PitchTransformError as error:
                record.update(status='pitch_transform_unusable',reason=str(error),waveform_metrics=metrics)
                for name in CONDITIONS[2:]:
                    path=audio_folder/f'{pair["pair_id"]}__{name}.wav'
                    path.unlink(missing_ok=True)
                    path.with_suffix('.tmp.wav').unlink(missing_ok=True)
                print(f'Excluded pair {pair["question_id"]}: {error}',flush=True)
            else:
                record.update(status='prepared',shifted=shifted,waveform_metrics=metrics,pilot_audit=audit)
        with (args.output/'pairs.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        cache[pair['pair_id']]=record
    usable=[p for p in pairs if cache[p['pair_id']]['status']=='prepared']
    if not usable:
        raise ValueError('No usable pitch pairs; inspect the pitch measurements before inference')
    listening=args.output/'listening';listening.mkdir(exist_ok=True)
    pages=[]
    for pair in usable[:3]:
        players=[]
        for style in CONDITIONS:
            source=pair[style]['audio_file'] if style in ('neutral','angry') else cache[pair['pair_id']]['shifted'][style]
            path=listening/f'{pair["pair_id"]}__{style}.wav'
            sf.write(path,load_audio(source,sf,librosa),16000,subtype='FLOAT')
            players.append(f'<p>{style}</p><audio controls preload="none" src="{path.name}"></audio>')
        pages.append(f'<h2>{pair["question_id"]}</h2>'+''.join(players))
    (listening/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pitch controls</title>'
        '<style>body{font:16px system-ui;max-width:900px;margin:30px auto}audio{width:100%}</style>'
        '<h1>Same questions: original and pitch-matched delivery</h1><p>Duration is unchanged; formant preservation is requested and RMS is matched to each source. '
        'Listen for pronunciation and processing artifacts. Automatic F0 checks do not establish naturalness.</p>'+''.join(pages))
    exclusions={}
    for pair in pairs:
        status=cache[pair['pair_id']]['status']
        if status!='prepared':exclusions[status]=exclusions.get(status,0)+1
    print(f'Pitch preparation: {len(usable)}/{len(pairs)} usable pairs; exclusions: {exclusions}.',flush=True)
    return usable,cache


@lru_cache(maxsize=None)
def original_answers(folder):
    return {r['audio_path']:r for r in read_rows(Path(folder)/'results-matched-accuracy.jsonl')}


def saved_originals(pair):
    result={}
    for style in ('neutral','angry'):
        record=pair[style];task=record['task']
        candidates=original_answers(record['source_folder'])
        row=candidates.get(task['audio_path'])
        if row and row.get('seed')==42 and row.get('prompt')==PROMPT and row.get('max_new_tokens')==512 and row.get('decoder')=='fast_dllm':
            if (row['question_id'],row['style'],row['voice'],str(row['reference_answer']))!=(pair['question_id'],style,pair['voice'],str(task['answer'])):
                raise ValueError('Saved original answer differs from recording plan')
            result[style]=row
    return result


def report(output):
    import numpy as np
    config=json.loads((output/'run_config.json').read_text())
    latest={(r['pair_id'],r['condition']):r for r in read_rows(output/'results.jsonl')}
    comparisons=[('neutral','angry'),('neutral','neutral_to_angry_pitch'),
                 ('angry','angry_to_neutral_pitch'),('neutral','angry_to_neutral_pitch'),
                 ('angry','neutral_to_angry_pitch')]
    summary=[]
    print('\nPitch controls: same question/voice. Test versus reference; strict scoring.',flush=True)
    print(f'{"Reference -> test":58} {"Pairs":>6} {"Ref acc":>8} {"Test acc":>8} {"Gain pp":>8} {"Wins":>5} {"Losses":>6} {"Exact p":>9}   95% CI gain pp')
    for reference,test in comparisons:
        pairs=[(latest[p['pair_id'],reference],latest[p['pair_id'],test]) for p in config['pairs']
               if (p['pair_id'],reference) in latest and (p['pair_id'],test) in latest]
        if not pairs:
            continue
        n=len(pairs);a=sum(r['correct'] for r,t in pairs);b=sum(t['correct'] for r,t in pairs)
        wins=sum(t['correct'] and not r['correct'] for r,t in pairs)
        losses=sum(r['correct'] and not t['correct'] for r,t in pairs)
        d=wins+losses;p=min(1,2*sum(math.comb(d,k) for k in range(min(wins,losses)+1))/2**d)
        draws=np.random.default_rng(42).multinomial(n,[wins/n,losses/n,(n-d)/n],size=200000)
        ci=np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist()
        summary.append(dict(reference=reference,test=test,pairs=n,ref_correct=a,test_correct=b,wins=wins,losses=losses,mcnemar_exact_p=p,ci95_gain_pp=ci))
        print(f'{reference+" -> "+test:58} {n:6} {a/n:8.1%} {b/n:8.1%} {100*(b-a)/n:+8.2f} {wins:5} {losses:6} {p:9.4f}   [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    print('Intervals and p-values are unadjusted. Pitch-usable subset only; processing can change more than F0.',flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,nargs='+',default=[Path('outputs/azure-jenny-next500'),Path('outputs/azure-jenny-remaining-test')])
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-pitch-controls'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.summary_only:
        report(args.output);return
    pairs,cache=prepare(args)
    if args.prepare_only:
        return
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import load_model,gen
    path=args.output/'results.jsonl'
    existing=read_rows(path,repair=True)
    if any(r.get('seed')!=42 or r.get('prompt')!=PROMPT for r in existing):
        raise ValueError('Cached pitch answers have a different recipe')
    complete={(r['pair_id'],r['condition']) for r in existing}
    originals={pair['pair_id']:saved_originals(pair) for pair in pairs}
    with path.open('a') as stream:
        for pair in pairs:
            for style,row in originals[pair['pair_id']].items():
                key=pair['pair_id'],style
                if key in complete:
                    continue
                value=dict(pair_id=pair['pair_id'],question_id=pair['question_id'],condition=style,
                           seed=42,prompt=PROMPT,response=row['response'],reused_from=pair[style]['source_folder'],
                           **score_response(row['response'],row['reference_answer']))
                stream.write(json.dumps(value)+'\n');complete.add(key)
    pending=sum((pair['pair_id'],name) not in complete for pair in pairs for name in CONDITIONS)
    print(f'{pending} pitch/control answers pending; compatible original answers reused.',flush=True)
    if not pending:
        report(args.output);return
    model,tokenizer,features=load_model()
    def generate(audio):
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return gen(audio,16000,model,tokenizer,features,PROMPT,512)
    # Check the restored model against a small fixed sample before trusting cached baselines.
    checks=[(pair,style,row) for pair in pairs for style,row in originals[pair['pair_id']].items()]
    for pair,style,row in checks[:5]:
        response=generate(load_audio(pair[style]['audio_file'],sf,librosa))
        old=score_response(row['response'],row['reference_answer']);fresh=score_response(response,row['reference_answer'])
        audit=dict(pair_id=pair['pair_id'],condition=style,old=old,fresh=fresh,response=response)
        with (args.output/'baseline-audit.jsonl').open('a') as stream:stream.write(json.dumps(audit)+'\n')
        if (old['correct'],old['predicted_answer'])!=(fresh['correct'],fresh['predicted_answer']):
            raise ValueError('Restored DIFFA baseline differs on the audit sample; inspect before using cached answers')
    for index,pair in enumerate(pairs,1):
        for name in CONDITIONS:
            if (pair['pair_id'],name) in complete:
                continue
            file=pair[name]['audio_file'] if name in ('neutral','angry') else cache[pair['pair_id']]['shifted'][name]
            audio=load_audio(file,sf,librosa)
            print(f'[{index}/{len(pairs)}] {pair["question_id"]} {name}',flush=True)
            started=time.monotonic();response=generate(audio)
            row=dict(pair_id=pair['pair_id'],question_id=pair['question_id'],voice=pair['voice'],condition=name,
                     seed=42,prompt=PROMPT,response=response,audio_file=file,generation_seconds=time.monotonic()-started,
                     **score_response(response,pair['neutral']['task']['answer']))
            with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
            complete.add((pair['pair_id'],name))
    report(args.output)


if __name__=='__main__':
    main()
