"""Wait for classification or steering processes in this repository to exit."""

import argparse
import os
from pathlib import Path
import time


NAMES={
    'phi-layer-nightly-inputs':{'run_phi4_layers_then_prepare.sh','run_phi4_all_layers_split.sh',
        'run_phi4_all_layer_diagnostic.sh','phi4_all_layer_diagnostic.py','prepare_phi4_next_tasks.sh'},
    'winning-inputs':{'run_missing_experiments.sh','run_reasoning_transfer.py',
                      'run_multiarith_mean.py','steer_ifeval_mean.py','run_spoken_ifeval.py',
                      'run_spoken_ifeval.sh','steer_bbh_mean.py','run_bbh_five_mean.py',
                      'run_gsm_strength_sweep.py'},
    'classification':{'classify_dataset_emotions.py','classify_cloud_smoke.py',
                      'run_dataset_emotions.sh','run_speeds_and_classification.sh'},
    'steering':{'run_jenny_promoted_third.py','evaluate_audio_steering.py','steer_audio.py',
                'run_large_steering.py','run_large_steering.sh'},
    'azure-emotions':{'run_emotions_after_steering.sh','run_jenny_third_accuracy.sh',
                      'classify_cloud_smoke.py','compare_cloud_accuracy.py'},
    'finalist-queue':{'finish_jenny_finalists.py','run_jenny_late_layers.py',
                      'evaluate_audio_steering.py','steer_audio.py','collect_temporal_vectors.py'},
    'audio-models':{'evaluate_qwen_omni.py','evaluate_kimi_audio.py','setup_qwen_omni.sh',
                    'setup_kimi_audio.sh','run_azure_after_classification.sh','run_azure_when_uploaded.py'},
    'pitch-speed':{'run_pitch_speed_after_models.py','evaluate_audio_pitch.py','evaluate_audio_speed.py'},
    'commonsense':{'evaluate_commonsense_audio.py'},
    'svamp':{'run_svamp500.py'},
    'bbh':{'run_bbh_logical200.py'},
    'ifeval':{'run_spoken_ifeval.py','run_spoken_ifeval.sh'},
    'ifbench':{'run_spoken_ifbench.py','run_ifbench_after_ifeval.sh'},
    'reasoning-transfer':{'run_reasoning_transfer.py'},
    'mean-steering':{'run_bbh_five_mean.py','steer_ifeval_mean.py','run_bbh_tracking250.py'},
    'strength-sweep':{'run_gsm_strength_sweep.py'},
    'seven-object':{'run_bbh_seven_mean.py'},
    'bbh-audio-models':{'run_bbh_moss_then_kimi.sh', 'evaluate_bbh_audio_models.py', 'setup_moss_audio.sh'},
}


def active_jobs(kind, root, proc=Path('/proc')):
    active=[]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            if (entry/'cwd').resolve()!=root.resolve():
                continue
            state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state in ('Z','X'):
                continue
            args=[a.decode(errors='replace') for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
            names={Path(a).name for a in args}
            matches=bool(names & NAMES[kind])
            if kind=='audio-models' and Path(args[0]).name in ('bash','sh') and '-c' in args:
                body=args[args.index('-c')+1]
                matches |= all(token in body for token in ('setup_qwen_omni.sh','setup_kimi_audio.sh','run_azure_followups.py'))
            if 'run_azure_followups.py' in names and '--stage' in args:
                stages={'classification':('classify',),'steering':('steering',),
                        'azure-emotions':('classify','diffa'),'finalist-queue':('steering',),
                        'audio-models':('qwen','kimi'),'pitch-speed':(),'commonsense':(),'svamp':(),'bbh':(),'ifeval':(),'ifbench':(),'reasoning-transfer':(),'mean-steering':(),'strength-sweep':()}
                matches |= args[args.index('--stage')+1] in stages.get(kind, ())
            if matches:
                active.append(int(entry.name))
        except (OSError,IndexError):
            continue
    return sorted(active)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=NAMES)
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args()
    if args.poll_seconds<=0:
        p.error('--poll-seconds must be positive')
    while True:
        pids=active_jobs(args.kind,Path.cwd())
        if not pids:
            print(f'No active {args.kind} jobs in this repository; continuing.',flush=True)
            return
        print(f'Waiting for {args.kind} PID(s): {pids}',flush=True)
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
