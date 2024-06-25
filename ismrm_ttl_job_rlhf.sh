#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40000M
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# The above comments are used by SLURM to set the job parameters.

set -e

# Set this to 0 if running on a cluster node.
islocal=0
RUN_OFFLINE=0

# Expriment parameters
EXPNAME="TrackToLearnRLHF"
COMETPROJECT="TrackToLearnRLHF"
EXPID="10-TractometerReward_Beluga_"_$(date +"%F-%H_%M_%S")
RLHFINTERNPV=20         # Number of seeds per tractogram generated during the RLHF pipeline
MAXEP=10                # Number of RLHF iterations
ORACLENBSTEPS=10        # Number of steps for the oracle
AGENTNBSTEPS=100        # Number of steps for the agent
PRETRAINSTEPS=1000      # Number of steps for pretraining if no agent checkpoint is provided.

NPV=8
SEEDS=(1111)
BATCHSIZE=4096
GAMMA=0.95
LR=0.0005
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
    ORACLECHECKPOINT=custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt
    AGENTCHECKPOINT=/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/experiments/TrackToLearnRLHF/1-Pretrain-AntoineOracle-Finetune_2024-06-09-20_55_13/1111/model
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

    echo "Unpacking datasets..."
    tar xf ~/projects/def-pmjodoin/levj1404/datasets/ismrm2015_2mm_ttl.tar.gz -C $DATADIR
    DATASETDIR=$DATADIR/ismrm2015_2mm

    echo "Copying oracle checkpoint..."
    cp ~/projects/def-pmjodoin/levj1404/oracles/ismrm_paper_oracle.ckpt $DATADIR
    
    echo "Copying agent checkpoint..."
    cp ~/projects/def-pmjodoin/levj1404/agents/1-Pretrain-AntoineOracle-Finetune_2024-06-09-20_55_13/* $DATADIR
    AGENTCHECKPOINT=~/projects/def-pmjodoin/levj1404/agents/1-Pretrain-AntoineOracle-Finetune_2024-06-09-20_55_13
fi

for RNGSEED in "${SEEDS[@]}"
do
    DEST_FOLDER="${EXPDIR}/${EXPNAME}/${EXPID}/${RNGSEED}"

    additionnal_args=()
    if [ $RUN_OFFLINE -eq 1 ]; then
        additionnal_args+=('--comet_offline_dir' "${LOGSDIR}")
    fi
    if [ -n "$AGENTCHECKPOINT" ]; then
        additionnal_args+=('--agent_checkpoint' "${AGENTCHECKPOINT}")
    fi

    # Start training
    ${PYTHONEXEC} -O $SOURCEDIR/TrackToLearn/trainers/rlhf_train.py \
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
        --replay_size 1000000 \
        --alignment_weighting 1.0 \
        --binary_stopping_threshold 0.1 \
        --oracle_checkpoint ${ORACLECHECKPOINT} \
        --oracle_validator \
        --oracle_stopping_criterion \
        --oracle_bonus 10.0 \
        --alignment_weighting 1.0 \
        --scoring_data "${DATASETDIR}/scoring_data" \
        --tractometer_reference "${DATASETDIR}/scoring_data/t1.nii.gz" \
        --tractometer_validator \
        --rng_seed ${RNGSEED} \
        --npv ${NPV} \
        --batch_size ${BATCHSIZE} \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --theta ${THETA} \
        --n_dirs 100 \
        --max_ep ${MAXEP} \
        --oracle_train_steps ${ORACLENBSTEPS} \
        --agent_train_steps ${AGENTNBSTEPS} \
        --rlhf_inter_npv ${RLHFINTERNPV} \
        --disable_oracle_training \
        --reward_with_gt \
        "${additionnal_args[@]}"
        # --dataset_to_augment "/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/ismrm_1mm_tracts_trainset_expandable.hdf5" \
        # --pretrain_max_ep ${PRETRAINSTEPS} \

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
