#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40000M
#SBATCH --time=40:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# The above comments are used by SLURM to set the job parameters.

set -e

# Set this to 0 if running on a cluster node.
islocal=1
RUN_OFFLINE=0

# Expriment parameters
EXPNAME="TrackToLearnPPO"
COMETPROJECT="TrackToLearnPPO"
EXPID="Classic_PPO_"_$(date +"%F-%H_%M_%S")
# RLHFINTERNPV=20         # Number of seeds per tractogram generated during the RLHF pipeline
MAXEP=1000              # Number of PPO iterations
# ORACLENBSTEPS=10        # Number of steps for the oracle
# AGENTNBSTEPS=100        # Number of steps for the agent
# PRETRAINSTEPS=1000      # Number of steps for pretraining if no agent checkpoint is provided.

NPV=2
SEEDS=(1111)
BATCHSIZE=4096
GAMMA=0.5
LR=0.00005
THETA=30

if [ $islocal -eq 1 ]; then
    # This script should be ran from the root of the project is ran locally.
    echo "Running training locally..."
    SOURCEDIR=.
    DATADIR=data/datasets/ismrm2015_2mm
    EXPDIR=data/experiments
    LOGSDIR=data/logs

    # If CONDAENV is not set, PYTHON EXEC should be python, else it should be the python executable of the conda environment.
    if [ -z $1 ]; then
        PYTHONEXEC=python
        echo "WARNING: No conda environment provided. Using the environment loaded when calling the script."
    else
        PYTHONEXEC=~/miniconda3/envs/$1/bin/python
    fi
    DATASETDIR=$DATADIR
    #ORACLECHECKPOINT=custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt
else
    echo "Running training on a cluster node..."
    module load python/3.10 cuda cudnn httpproxy
    SOURCEDIR=~/TrackToLearn
    DATADIR=$SLURM_TMPDIR/data
    EXPDIR=$SLURM_TMPDIR/experiments
    LOGSDIR=$SLURM_TMPDIR/logs
    PYTHONEXEC=python
    export COMET_API_KEY=$(cat ~/.comet_api_key)

    ORACLECHECKPOINT=$DATADIR/ismrm_paper_oracle.ckpt

    # Prepare virtualenv
    echo "Sourcing ENV-TTL-2 virtual environment..."
    source ~/ENV-TTL-2/bin/activate # Ideally, we would build the environnement within the node itself, but too much dependencies for now.

    # Prepare datasets
    mkdir $DATADIR
    mkdir $EXPDIR
    mkdir $LOGSDIR

    echo "Unpacking datasets..."
    tar xf ~/projects/def-pmjodoin/levj1404/datasets/ismrm2015_2mm_ttl.tar.gz -C $DATADIR
    DATASETDIR=$DATADIR/ismrm2015_2mm

    echo "Copying oracle checkpoint..."
    cp ~/projects/def-pmjodoin/levj1404/oracles/ismrm_paper_oracle.ckpt $DATADIR
fi

for RNGSEED in "${SEEDS[@]}"
do
    DEST_FOLDER="${EXPDIR}/${EXPNAME}/${EXPID}"

    additionnal_args=()
    # if [ $RUN_OFFLINE -eq 1 ]; then
    #     additionnal_args+=('--comet_offline_dir' "${LOGSDIR}")
    # fi
    # if [ -n "$AGENTCHECKPOINT" ]; then
    #     additionnal_args+=('--agent_checkpoint' "${AGENTCHECKPOINT}")
    # fi

    # Start training
    # Start training
    ${PYTHONEXEC} \
        $SOURCEDIR/TrackToLearn/trainers/classic_ppo_train.py \
        ${DEST_FOLDER} \
        "${COMETPROJECT}" \
        "${EXPID}" \
        "${DATASETDIR}/ismrm2015.hdf5" \
        --workspace "mrzarfir" \
        --hidden_dims "1024-1024-1024" \
        --use_comet \
        --n_actor 4096 \
        --min_length 20 \
        --max_length 200 \
        --noise 0.0 \
        --alignment_weighting 1.0 \
        --binary_stopping_threshold 0.1 \
        --oracle_bonus 0.0 \
        --alignment_weighting 1.0 \
        --scoring_data "${DATASETDIR}/scoring_data" \
        --tractometer_reference "${DATASETDIR}/scoring_data/t1.nii.gz" \
        --tractometer_validator \
        --rng_seed ${RNGSEED} \
        --npv ${NPV} \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --theta ${THETA} \
        --n_dirs 100 \
        --max_ep ${MAXEP} \
        --entropy_loss_coeff 0.001 \
        --lmbda 0.95 \
        --eps_clip 0.2 \
        --K_epochs 30 \
        --action_std 0.0 \
        "${additionnal_args[@]}"

done

if [ $islocal -eq 1 ]; then
    echo "Experiment results are saved in ${EXPDIR}."
    echo "To see the results on Comet.ml, please run \"comet upload ${LOGSDIR}/<comet-exp-hash>.zip\"."
    echo "Done."
else
    # Archive and save everything
    OUTNAME=${EXPNAME}$(date +"%F").tar

    echo "Archiving experiment..."
    tar -cvf ${DATADIR}/${OUTNAME} $EXPDIR $LOGSDIR
    echo "Copying archive to scratch..."
    cp ${DATADIR}/${OUTNAME} ~/scratch/${OUTNAME}
fi
