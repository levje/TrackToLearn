#!/bin/bash

python TrackToLearn/trainers/sac_auto_train.py \
    "data/experiments/test" \
    "test" \
    "NoSearch" \
    "data/datasets/fibercup/fibercup.hdf5" \
    --max_ep 5 \
    --oracle_checkpoint "models/oracle_fibercup_simple.ckpt" \
    --oracle_validator \
    --oracle_stopping_criterion \
    --oracle_bonus 10.0 \
    --scoring_data "data/datasets/fibercup/scoring_data" \
    --tractometer_reference "data/datasets/fibercup/masks/fibercup_wm.nii.gz" \
    --tractometer_validator \
    --use_comet \
    --workspace "mrzarfir" \
    --rng_seed 1111 \
    --n_actor 5000 \
    --npv 33 \
    --min_length 20 \
    --max_length 200 \
    --noise 0.0 \
    --batch_size 4096 \
    --replay_size 1000000
