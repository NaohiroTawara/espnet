#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
audio_format=sph

./sre.sh \
    --lang en \
    --fs 8000 \
    --skip_dumping true \
    --train_set train \
    --ngpu 5 \
    --nj 4 \
    --collect_feats true \
    --valid_set valid \
    --test_sets "test" \
    --apply_vad true \
    "$@"
