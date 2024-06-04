#!/bin/bash

python TrackToLearn/runners/ttl_track.py \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015/fodfs/ismrm2015_fodf.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015/masks/ismrm2015_interface.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015/masks/ismrm2015_mask_wm.nii.gz" \
    "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/results/ismrm2015_paper_oracle.trk" \
    --agent custom_models/ismrm_paper_oracle \
    --hyperparameters custom_models/ismrm_paper_oracle/hyperparameters.json \
    --min_length 20 \
    --max_length 200 \
    --npv 10 \
    --n_actor 5000 \
    --rng_seed 1111 \
    --fa_map "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015/dti/ismrm2015_fa.nii.gz"
