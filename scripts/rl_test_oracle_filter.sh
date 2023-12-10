#!/usr/bin/env bash

set -e

# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Should be relatively stable
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data

# Data params
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/anat/${SUBJECT_ID}_T1.nii.gz

n_actor=50000
npv=20
min_length=20
max_length=200
prob=1.0

EXPERIMENT=$1
ID=$2

subjectids=(ismrm2015)
seeds=(1111 2222 3333 4444 5555)

for SEED in "${seeds[@]}"
do
  for SUBJECT_ID in "${subjectids[@]}"
  do
    EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
    SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data
    DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

    dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
    reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/anat/${SUBJECT_ID}_T1.nii.gz
    filename=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".tck

    ttl_validation.py \
      "$DEST_FOLDER" \
      "$EXPERIMENT" \
      "$ID" \
      "${dataset_file}" \
      "${SUBJECT_ID}" \
      "${reference_file}" \
      $DEST_FOLDER/model \
      $DEST_FOLDER/model/hyperparameters.json \
      ${DEST_FOLDER}/${filename} \
      --prob="${prob}" \
      --npv="${npv}" \
      --n_actor="${n_actor}" \
      --min_length="$min_length" \
      --max_length="$max_length" \
      --use_gpu \
      --binary_stopping_threshold=0.1 \
      --oracle_validator \
      --oracle_filter \
      --oracle_checkpoint='epoch_39_ismrm2015v3.ckpt' \

    validation_folder=$DEST_FOLDER/scoring_"${SUBJECT_ID}"_${npv}_filtered

    mkdir -p $validation_folder

    mv $DEST_FOLDER/${filename} $validation_folder/

    if [[ -d ${validation_folder}/scoring ]]; then
      rm -r $validation_folder/scoring
    fi
    ./scripts/tractometer.sh $validation_folder/${filename} $validation_folder/scoring ./

  done
done