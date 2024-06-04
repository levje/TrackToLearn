#!/bin/bash

python TrackToLearn/trainers/sac_auto_train.py \
    "data/experiments/ISMRM-TractOracleRL-NewOracle-Step075" \
    "ISMRM-TractOracleRL-NewOracle-Step075" \
    "Step075" \
    "data/datasets/ismrm2015/ismrm2015.hdf5" \
    --max_ep 1000 \
    --oracle_checkpoint custom_models/ismrm2015_new_oracle_step075/ismrm2015_new_oracle_step075.ckpt \
    --oracle_validator \
    --oracle_stopping_criterion \
    --oracle_bonus 10.0 \
    --scoring_data "data/datasets/ismrm2015/scoring_data" \
    --tractometer_reference "data/datasets/ismrm2015/anat/ismrm2015_T1.nii.gz" \
    --tractometer_validator \
    --use_comet \
    --workspace "mrzarfir" \
    --rng_seed 1111 \
    --n_actor 4096 \
    --npv 4 \
    --min_length 20 \
    --max_length 200 \
    --noise 0.0 \
    --batch_size 4096 \
    --replay_size 1000000 \
    --lr 0.0005 
