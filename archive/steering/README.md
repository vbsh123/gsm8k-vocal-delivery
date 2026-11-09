# Historical extraction and steering experiments

These files preserve the old experiment selections and settings. Active shared generation is in `steer_audio.py`; hooks and reporting are in `steer.py`. New extraction is in `extract_audio_vectors.py`.

Run archived Python scripts from the repository root using `python -m archive.steering.SCRIPT_NAME`, without `.py`. Run archived shell launchers with `bash archive/steering/SCRIPT_NAME.sh`.

| Experiment | Script |
|---|---|
| Original angry-minus-neutral extraction | `extract_jenny_angry.py` |
| Independent all-forwards extraction, original standalone copy | `extract_jenny_angry_all_forwards.py` |
| Original SVD/probe collection and fitting | `extract_jenny_svd_probe.py` |
| Initial 50-recording sweep | `steer_jenny_angry.py` |
| Six-question plan adapter; also accepts prepared larger plans | `steer_jenny_six.py` |
| Original 74-question recovery | `steer_jenny_recovery.py` |
| Expanded conditions on the same74 | `steer_jenny_recovery_followup.py` |
| Both-correct regression cohort | `steer_jenny_regressions.py` |
| Complete L11/L12 on the full460 | `steer_jenny_complete.py` |
| Ten-question screen | `steer_jenny_recovery10.py` |
| Promoted ×1.25 conditions on74 | `steer_jenny_recovery74_promoted.py` |
| L16/L23 strength and regression follow-up | `steer_jenny_L16_L23_followup.py` |
| L16 full-cohort completion, including `--l16-only` | `steer_jenny_compare_L12_L16.py` |
| Original Spoken GSM transfer | `steer_jenny_original300.py` |
| Other generated deliveries | `steer_jenny_deliveries.py` |
| Historical CLI using per-recording derived seeds | `steer.py` |
| Earlier strength/combination plans | `run_steering_followup.py` |

The historical CLI's base seed42 is hashed with the clip ID. The newer Jenny runner resets literal42. Moving these files does not change either policy or convert old saved results.
