#!/usr/bin/env bash

set -e

# This should point to your dataset folder
DATASET_FOLDER=${TRACK_TO_LEARN_DATA}

# Should be relatively stable
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data

# Data params
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

n_actor=50000
npv=33
min_length=20
max_length=200

list_n_rollouts=(2)
list_backup_size=(1)
list_extra_n_steps=(5)
list_roll_n_steps=(1)

EXPERIMENT=$1
ID=$2

validstds=(1.0)
subjectids=(fibercup)
seeds=(1111)

for SEED in "${seeds[@]}"
do
  for SUBJECT_ID in "${subjectids[@]}"
  do
    for prob in "${validstds[@]}"
    do
      EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
      SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data
      DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

      dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
      reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz
      filename=tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".tck

      for n_rollouts in "${list_n_rollouts[@]}"
      do
	for backup_size in "${list_backup_size[@]}"
	do
	  for extra_n_steps in "${list_extra_n_steps[@]}"
	  do
	    for roll_n_steps in "${list_roll_n_steps[@]}"
	    do

	     echo "Starting validation with n-rollouts: $n_rollouts, backup_size: $backup_size, extra_n_steps: $extra_n_steps, roll_n_steps: $roll_n_steps"

      	     echo $DEST_FOLDER/model/hyperparameters.json
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
             --fa_map="$DATASET_FOLDER"/datasets/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
	     --dense_oracle_weighting=1.0 \
	     --oracle_checkpoint="epoch_49_fibercup_transformer.ckpt"

	     validation_folder=$DEST_FOLDER/scoring_"${prob}"_"${SUBJECT_ID}"_${npv}

             mkdir -p $validation_folder

      	     mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".tck $validation_folder/

      	     outdir=$validation_folder/ref_scoring_dense_oracle_w1

      	     if [[ -d $outdir ]]; then
         	rm -r $outdir
      	     fi

	     ./scripts/tractometer_fibercup.sh $validation_folder/${filename} $outdir $SCORING_DATA
     	         done
      	      done
 	   done
 	done
     done
  done
done
