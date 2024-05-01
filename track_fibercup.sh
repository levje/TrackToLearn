#!/bin/bash

python TrackToLearn/runners/ttl_track.py \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/fibercup/fodfs/fibercup_fodf.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/fibercup/maps/interface.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/fibercup/masks/fibercup_wm.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/results/fibercup_simple_oracle_tractogram_noisy_1000.trk" \
    --agent custom_models/fibercup_simple_oracle \
    --hyperparameters custom_models/fibercup_simple_oracle/hyperparameters.json \
    --min_length 20 \
    --max_length 200 \
    --npv 1000 \
    --n_actor 5000 \
    --rng_seed 1111 \
    --fa_map "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/fibercup/dti/fibercup_fa.nii.gz" \
    --noise 0.1
