#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# dlhpcstarter \
#     -t cxrmate_ed \
#     -c config/stage_1.yaml \
#     --train \
#     --test \
#     --max_epochs 1 \
#     --limit-train-batches 0.0001 \
#     --limit-val-batches 0.01 \
#     --limit-test-batches 0.005 \
#     --stages_module stages \
#     --exp_dir /scratch3/nic261/test/experiments \
#     --database_dir /scratch3/nic261/database/cxrmate_ed
    
# dlhpcstarter \
#     -t cxrmate_ed \
#     -c config/stage_2.yaml \
#     --train \
#     --test \
#     --max_epochs 1 \
#     --limit-train-batches 0.0002 \
#     --limit-val-batches 0.015 \
#     --limit-test-batches 0.01 \
#     --stages_module stages \
#     --exp_dir /scratch3/nic261/test/experiments \
#     --database_dir /scratch3/nic261/database/cxrmate_ed \
#     --warm_start_other_exp_trial_dir /scratch3/nic261/test/experiments/cxrmate_ed/stage_1/trial_0

dlhpcstarter \
    -t cxrmate_ed \
    -c config/stage_3.yaml \
    --train \
    --test \
    --max_epochs 1 \
    --limit-train-batches 0.0005 \
    --limit-val-batches 0.06 \
    --limit-test-batches 0.04 \
    --stages_module stages \
    --exp_dir /scratch3/nic261/test/experiments \
    --database_dir /scratch3/nic261/database/cxrmate_ed \
    --warm_start_other_exp_trial_dir /scratch3/nic261/test/experiments/cxrmate_ed/stage_2/trial_0 \
    --devices 1
