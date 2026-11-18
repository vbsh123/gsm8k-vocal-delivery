#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo 'Phi MultiArith L17: existing290-pair vector, same290 tests, -1.5 and -2; no collection.'
.venv-bbh-phi4/bin/python -u phi4_reverse_followups.py --stage multiarith-extra --layer 17 --concurrent \
  >outputs/phi4-multiarith-L17-extra.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_reverse_followups.py \
  --stage summary-multiarith --layer 17 --output outputs/phi4-multiarith-reverse290-290-L17-extra-minus1.5-minus2 \
  >>outputs/phi4-multiarith-L17-extra.log 2>&1
echo 'Phi GSM happy400 follows MultiArith; neutral baselines reused where compatible.'
.venv-bbh-phi4/bin/python -u evaluate_gsm_happy400.py --mode evaluate --model phi4 --concurrent \
  >outputs/phi4-gsm-happy400.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode summary --model phi4 \
  >>outputs/phi4-gsm-happy400.log 2>&1
