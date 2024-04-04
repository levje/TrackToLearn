import os
import shutil
import subprocess
from datetime import datetime

# Set up environment variables
TRACK_TO_LEARN_DATA = os.getenv('TRACK_TO_LEARN_DATA', '')
LOCAL_TRACK_TO_LEARN_DATA = os.getenv('LOCAL_TRACK_TO_LEARN_DATA', '')

DATASET_FOLDER = os.path.join(TRACK_TO_LEARN_DATA, '')
WORK_DATASET_FOLDER = os.path.join(LOCAL_TRACK_TO_LEARN_DATA, '')

VALIDATION_DATASET_NAME = 'fibercup'
DATASET_NAME = 'fibercup'
EXPERIMENTS_FOLDER = os.path.join(DATASET_FOLDER, 'experiments')
WORK_EXPERIMENTS_FOLDER = os.path.join(WORK_DATASET_FOLDER, 'experiments')
SCORING_DATA = os.path.join(WORK_DATASET_FOLDER, 'datasets', VALIDATION_DATASET_NAME, 'scoring_data')

#os.makedirs(os.path.join(WORK_DATASET_FOLDER, 'datasets', DATASET_NAME), exist_ok=True)

print("Transfering data to working folder...")
#shutil.copytree(os.path.join(DATASET_FOLDER, 'datasets', VALIDATION_DATASET_NAME),
#                os.path.join(WORK_DATASET_FOLDER, 'datasets', VALIDATION_DATASET_NAME))
#shutil.copytree(os.path.join(DATASET_FOLDER, 'datasets', DATASET_NAME),
#                os.path.join(WORK_DATASET_FOLDER, 'datasets', DATASET_NAME))

dataset_file = os.path.join(WORK_DATASET_FOLDER, 'datasets', DATASET_NAME, DATASET_NAME + '.hdf5')
validation_dataset_file = os.path.join(WORK_DATASET_FOLDER, 'datasets', VALIDATION_DATASET_NAME, VALIDATION_DATASET_NAME + '.hdf5')
reference_file = os.path.join(WORK_DATASET_FOLDER, 'datasets', VALIDATION_DATASET_NAME, 'masks', VALIDATION_DATASET_NAME + '_wm.nii.gz')

# RL params
max_ep = 1000
log_interval = 50
lr = 0.0005
gamma = 0.95

# Model params
prob = 1.0

# Env parameters
npv = 33
theta = 30
n_actor = 4096

EXPERIMENT = 'SAC_Auto_Rollouts_FiberCupTrainOracle'
ID = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

seeds = [1111]

for rng_seed in seeds:
    DEST_FOLDER = os.path.join(WORK_EXPERIMENTS_FOLDER, EXPERIMENT, ID, str(rng_seed))

    completed_process = subprocess.run(['python', 'TrackToLearn/trainers/sac_auto_train.py',
                    DEST_FOLDER,
                    EXPERIMENT,
                    ID,
                    dataset_file,
                    '--max_ep=' + str(max_ep),
                    '--log_interval=' + str(log_interval),
                    '--lr=' + str(lr),
                    '--gamma=' + str(gamma),
                    '--rng_seed=' + str(rng_seed),
                    '--npv=' + str(npv),
                    '--theta=' + str(theta),
                    '--alignment_weighting=1.0',
                    '--hidden_dims=1024-1024-1024',
                    '--n_dirs=100',
                    '--n_actor=' + str(n_actor),
                    '--action_type=cartesian',
                    '--interface_seeding',
                    '--prob=' + str(prob),
                    '--use_gpu',
                    '--use_comet',
                    '--binary_stopping_threshold=0.1',
                    '--coverage_weighting=0.0',
                    '--tractometer_validator',
                    '--tractometer_dilate=3',
                    '--tractometer_reference=' + reference_file,
                    '--scoring_data=' + SCORING_DATA,
                    '--oracle_validator',
                    '--sparse_oracle_weighting=10.0',
                    '--oracle_stopping',
                    '--oracle_checkpoint=epoch_49_fibercup_transformer.ckpt',
                    '--workspace=mrzarfir',
                    '--do_rollout'
                    ])

    if completed_process.returncode != 0:
        exit(completed_process.returncode)

    os.makedirs(os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT, ID), exist_ok=True)
    shutil.copytree(DEST_FOLDER, os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT, ID, str(rng_seed)))
