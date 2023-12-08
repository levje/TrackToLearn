import os
import sys
import subprocess
import shutil

SEARCH_ID = "1"

# Function to execute a command
def run_command(command):
    subprocess.run(command, shell=True)

# Set dataset folder (adjust this as per your environment)
DATASET_FOLDER = os.environ.get('TRACK_TO_LEARN_DATA')

# Parameters
n_actor = 50000
npv = 33
min_length = 20
max_length = 200

list_n_rollouts = [5]
list_backup_size = [5]
list_extra_n_steps = [5]
list_roll_n_steps = [1]

EXPERIMENT = "SAC_Auto_FiberCupTrainOracle"
ID = "2023-11-23-21_34_57"

prob = 1.0
SUBJECT_ID = 'fibercup'
SEED = 1111


EXPERIMENTS_FOLDER = os.path.join(DATASET_FOLDER, 'experiments')
SCORING_DATA = os.path.join(DATASET_FOLDER, 'datasets', SUBJECT_ID, 'scoring_data')
DEST_FOLDER = os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT, ID, str(SEED))

dataset_file = os.path.join(DATASET_FOLDER, 'datasets', SUBJECT_ID, f'{SUBJECT_ID}.hdf5')
reference_file = os.path.join(DATASET_FOLDER, 'datasets', SUBJECT_ID, 'masks', f'{SUBJECT_ID}_wm.nii.gz')
filename = f'tractogram_{EXPERIMENT}_{ID}_{SUBJECT_ID}.tck'

for n_rollouts in list_n_rollouts:
    for backup_size in list_backup_size:
        for extra_n_steps in list_extra_n_steps:
            for roll_n_steps in list_roll_n_steps:
                print(f"Starting validation with n-rollouts for search {SEARCH_ID}: {n_rollouts}, backup_size: {backup_size}, extra_n_steps: {extra_n_steps}, roll_n_steps: {roll_n_steps}")

                # Execute ttl_validation.py command
                validation_cmd = f"ttl_validation.py {DEST_FOLDER} {EXPERIMENT} {ID} {dataset_file} {SUBJECT_ID} {reference_file} {DEST_FOLDER}/model {DEST_FOLDER}/model/hyperparameters.json {DEST_FOLDER}/{filename} --prob={prob} --npv={npv} --n_actor={n_actor} --min_length={min_length} --max_length={max_length} --use_gpu --binary_stopping_threshold=0.1 --fa_map={DATASET_FOLDER}/datasets/{SUBJECT_ID}/dti/{SUBJECT_ID}_fa.nii.gz --scoring_data={SCORING_DATA} --do_rollout --n_rollouts={n_rollouts} --backup_size={backup_size} --extra_n_steps={extra_n_steps} --roll_n_steps={roll_n_steps} --dense_oracle_weighting=1.0 --oracle_checkpoint='epoch_49_fibercup_transformer.ckpt'"
                run_command(validation_cmd)

                validation_folder = os.path.join(DEST_FOLDER, f'scoring_{prob}_{SUBJECT_ID}_{npv}')
                os.makedirs(validation_folder, exist_ok=True)

                # Move tractogram file
                tractogram_filename = f'tractogram_{EXPERIMENT}_{ID}_{SUBJECT_ID}.tck'
                tractogram_file = os.path.join(DEST_FOLDER, tractogram_filename)
                if os.path.exists(tractogram_file):
                    os.rename(tractogram_file, os.path.join(validation_folder, filename))

                outdir = os.path.join(validation_folder, f'{SEARCH_ID}-search_scoring_oracle_{n_rollouts}r_{backup_size}b_{extra_n_steps}x_{roll_n_steps}n')

                # Remove existing directory if it exists
                if os.path.isdir(outdir):
                    shutil.rmtree(outdir)

                # Execute tractometer_fibercup.sh command
                tractometer_cmd = f"./scripts/tractometer_fibercup.sh {validation_folder}/{filename} {outdir} {SCORING_DATA}"
                run_command(tractometer_cmd)

                shutil.copy(os.path.join(validation_folder, filename), os.path.join(outdir, tractogram_filename))
                print(f"Copying tractogram file to {validation_folder}/{filename}")
