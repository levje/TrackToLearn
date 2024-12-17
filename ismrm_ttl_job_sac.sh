#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22000M
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# The above comments are used by SLURM to set the job parameters.
set -e

# Expriment parameters
EXPNAME="TrackToLearn"
COMETPROJECT="TrackToLearn"
NB_STREAMLINES_POINTS=32
EXPID="ConvPolicy-radius2-"_$(date +"%F-%H_%M_%S")
MAXEP=1000
BATCHSIZE=4096
SEEDS=(1111)
NPV=20
GAMMA=0.95
LR=0.0002
THETA=30

# Check if the script is ran locally or on a cluster node.
if [ -z $SLURM_JOB_ID ]; then
    islocal=1
else
    islocal=0
fi

if [ $islocal -eq 1 ]; then
    # This script should be ran from the root of the project is ran locally.
    echo "Running training locally..."
    SOURCEDIR=.
    DATADIR=data/datasets/ismrm2015_2mm
    EXPDIR=data/experiments
    LOGSDIR=data/logs
    # BACKUPDIR=data/backups

    # If CONDAENV is not set, PYTHON EXEC should be python, else it should be the python executable of the conda environment.
    if [ -z $1 ]; then
        PYTHONEXEC=python
        echo "WARNING: No conda environment provided. Using the environment loaded when calling the script."
    else
        PYTHONEXEC=~/miniconda3/envs/$1/bin/python
    fi
    DATASETDIR=$DATADIR

    # Oracle Antoine with partial streamlines (dense).
    ORACLE_CRIT_CHECKPOINT=custom_models/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Dense/_best_vc_epoch.ckpt

    # Oracle trained on full streamlines (not dense).
    ORACLE_REWARD_CHECKPOINT=custom_models/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Classif/_best_vc_epoch.ckpt

    # THIS IS THE CHECKPOINT WE ARE CURRENTLY USING.
    AGENTCHECKPOINT=custom_models/sac_checkpoint/model/last_model_state.ckpt

else
    echo "Running training on a cluster node..."
    module load python/3.10 cuda cudnn httpproxy
    SOURCEDIR=~/TrackToLearn
    DATADIR=$SLURM_TMPDIR/data
    EXPDIR=$SLURM_TMPDIR/experiments
    LOGSDIR=$SLURM_TMPDIR/logs
    PYTHONEXEC=python
    PROJECTS_DIR=~/projects/def-pmjodoin/levje
    export COMET_API_KEY=$(cat ~/.comet_api_key)

    # Prepare virtualenv
    echo "Sourcing ENV-TTL-2 virtual environment..."
    source ~/ENV-TTL-2/bin/activate # Ideally, we would build the environnement within the node itself, but too much dependencies for now.

    # Prepare datasets
    mkdir -p $DATADIR
    mkdir -p $EXPDIR

    echo "Unpacking datasets..."
    tar xf ${PROJECTS_DIR}/datasets/ismrm2015_2mm_ttl.tar.gz -C $DATADIR
    DATASETDIR=$DATADIR/ismrm2015_2mm

    echo "Copying oracle checkpoint..."
    cp ${PROJECTS_DIR}/oracles/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Classif/_best_vc_epoch.ckpt $DATADIR/oracle_reward.ckpt
    cp ${PROJECTS_DIR}/oracles/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Dense/_best_vc_epoch.ckpt $DATADIR/oracle_crit.ckpt

    ORACLE_REWARD_CHECKPOINT=$DATADIR/oracle_reward.ckpt
    ORACLE_CRIT_CHECKPOINT=$DATADIR/oracle_crit.ckpt
fi

for RNGSEED in "${SEEDS[@]}"
do
    DEST_FOLDER="${EXPDIR}/${EXPNAME}/${EXPID}/${RNGSEED}"

    # Append the current seed to the EXPID
    EXPID="${RNGSEED}-${EXPID}"

    additionnal_args=()

    if [[ -n "${BACKUPDIR}" ]]; then
        additionnal_args+=('--backup_dir' "${BACKUPDIR}")
    fi

    # Start training
    python -O $SOURCEDIR/TrackToLearn/trainers/sac_auto_train.py \
        ${DEST_FOLDER} \
        "${COMETPROJECT}" \
        "${EXPID}" \
        "${DATASETDIR}/ismrm2015.hdf5" \
        --max_ep ${MAXEP} \
        --hidden_dims "1024-1024-1024" \
        --oracle_reward_checkpoint ${ORACLE_REWARD_CHECKPOINT} \
        --oracle_crit_checkpoint ${ORACLE_CRIT_CHECKPOINT} \
        --oracle_validator \
        --oracle_stopping_criterion \
        --oracle_bonus 10.0 \
        --scoring_data "${DATASETDIR}/scoring_data" \
        --tractometer_reference "${DATASETDIR}/scoring_data/t1.nii.gz" \
        --tractometer_validator \
        --use_comet \
        --workspace "mrzarfir" \
        --rng_seed ${RNGSEED} \
        --n_actor 4096 \
        --npv ${NPV} \
        --min_length 20 \
        --max_length 200 \
        --noise 0.0 \
        --batch_size ${BATCHSIZE} \
        --replay_size 100000 \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --theta ${THETA} \
        --alignment_weighting 1.0 \
        --binary_stopping_threshold 0.1 \
        --n_dirs=100 \
        --alignment_weighting=1.0 \
        "${additionnal_args[@]}"

    # POST-PROCESSING
    bash scripts/tractogram_post_processing.sh ${DEST_FOLDER} ${DATASETDIR}
done

if [ $islocal -eq 1 ]; then
    echo "Experiment results are saved in ${EXPDIR}."
    echo "To see the results on Comet.ml, please run \"comet upload ${LOGSDIR}/<comet-exp-hash>.zip\"."
    echo "Done."
else
    # Archive and save everything
    OUTNAME=${EXPID}$(date -d "today" +"%Y%m%d%H%M").tar

    echo "Archiving experiment..."
    tar -cvf ${DATADIR}/${OUTNAME} $EXPDIR $LOGSDIR
    echo "Copying archive to scratch..."
    cp ${DATADIR}/${OUTNAME} ~/scratch/${OUTNAME}
fi
