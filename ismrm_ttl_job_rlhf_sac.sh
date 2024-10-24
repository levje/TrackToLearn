#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22000M
#SBATCH --time=0-25:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# The above comments are used by SLURM to set the job parameters.
set -e

# Expriment parameters
EXPNAME="TrackToLearnRLHF"
COMETPROJECT="TrackToLearnRLHF"
EXPID="Huge-SAC-SFT+RLHF-"_$(date +"%F-%H_%M_%S")
ALG="SACAuto"
RLHFINTERNPV=30         # Number of seeds per tractogram generated during the RLHF pipeline
MAXEP=100                # Number of RLHF iterations
ORACLENBSTEPS=3         # Number of steps for the oracle
AGENTNBSTEPS=50         # Number of steps for the agent
PRETRAINSTEPS=1000      # Number of steps for pretraining if no agent checkpoint is provided.

NPV=20 # Number of points per tractogram for training
SEEDS=(1111)
BATCHSIZE=4096
GAMMA=0.95 # Reward discounting (could also be 0.95)
LR=0.0005 # 1e-5
THETA=30

# Oracle training params
ORACLE_LR=0.00005 # This will override the LR within the checkpoint.
TOTAL_BATCH_SIZE=2048
ORACLE_MICRO_BATCH_SIZE=1024
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / ORACLE_MICRO_BATCH_SIZE))
DISABLE_ORACLE_TRAINING=0

# PPO hparams
ENTROPY_LOSS_COEFF=0.0001 # Entropy bonus for policy loss
ACTION_STD=0.0 # Std use for the action
K_EPOCHS=30
LAMBDA=0.95 # For advantage estimation
POLICYCLIP=0.1
VALUECLIP=0.1
KL_PENALTY_COEFF=0.02
KL_TARGET=0.005
KL_HORIZON=1000
ADAPTIVE_KL=0 # Set to 1 to use adaptive KL
INIT_CRITIC_TO_ORACLE=0 # Set to 1 to initialize the critic to the oracle

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

    # If CONDAENV is not set, PYTHON EXEC should be python, else it should be the python executable of the conda environment.
    if [ -z $1 ]; then
        PYTHONEXEC=python
        echo "WARNING: No conda environment provided. Using the environment loaded when calling the script."
    else
        PYTHONEXEC=~/miniconda3/envs/$1/bin/python
    fi
    DATASETDIR=$DATADIR
    
    # Oracle Antoine with partial streamlines (dense).
    ORACLE_CRIT_CHECKPOINT=custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt

    # Oracle trained on full streamlines (not dense).
    ORACLE_REWARD_CHECKPOINT=custom_models/ismrm_classif_oracle/ismrm_classif_oracle.ckpt

    # THIS IS THE CHECKPOINT WE ARE CURRENTLY USING.
    # AGENTCHECKPOINT=custom_models/sac_checkpoint/model/last_model_state.ckpt

    # Setup dataset to augment
    DATASET_TO_AUGMENT= # data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine.hdf5
else
    echo "Running training on a cluster node..."
    module load python/3.10 cuda cudnn httpproxy
    SOURCEDIR=~/TrackToLearn
    DATADIR=$SLURM_TMPDIR/data
    EXPDIR=$SLURM_TMPDIR/experiments
    PYTHONEXEC=python
    PROJECTS_DIR=~/projects/def-pmjodoin/levje
    export COMET_API_KEY=$(cat ~/.comet_api_key)

    # Prepare virtualenv
    echo "Sourcing ENV-TTL-2 virtual environment..."
    source ~/ENV-TTL-2/bin/activate

    # Prepare datasets
    mkdir -p $DATADIR
    mkdir -p $EXPDIR

    echo "Unpacking datasets..."
    tar xf ${PROJECTS_DIR}/datasets/ismrm2015_2mm_ttl.tar.gz -C $DATADIR
    DATASETDIR=$DATADIR/ismrm2015_2mm

    echo "Copying oracle checkpoint..."
    cp ${PROJECTS_DIR}/oracles/ismrm_paper_oracle.ckpt $DATADIR
    cp ${PROJECTS_DIR}/oracles/ismrm_classif_oracle.ckpt $DATADIR

    echo "Copying agent checkpoint..."
    cp ${PROJECTS_DIR}/agents/sac_checkpoint/* $DATADIR/sac_checkpoint
    AGENTCHECKPOINT=$DATADIR/sac_checkpoint/last_model_state.ckpt

    # Oracle Antoine with partial streamlines (dense).
    ORACLE_CRIT_CHECKPOINT=$DATADIR/ismrm_paper_oracle.ckpt
    
    # Oracle trained on full streamlines (not dense).
    ORACLE_REWARD_CHECKPOINT=$DATADIR/ismrm_classif_oracle.ckpt

    # Setup dataset to augment
    # cp ${PROJECTS_DIR}/datasets/train_test_classical_tracts_dataset.hdf5 $DATADIR
    $DATASET_TO_AUGMENT= #$DATADIR/train_test_classical_tracts_dataset.hdf5
fi

for RNGSEED in "${SEEDS[@]}"
do
    DEST_FOLDER="${EXPDIR}/${EXPNAME}/${EXPID}/${RNGSEED}"

    additionnal_args=()
    if [ -n "$AGENTCHECKPOINT" ]; then
        additionnal_args+=('--agent_checkpoint' "${AGENTCHECKPOINT}")
    else
        additionnal_args+=('--pretrain_max_ep' "${PRETRAINSTEPS}")
    fi

    # If ORACLE_LR is set AND is higher than zero, add it to the arguments.
    if [[ -n "${ORACLE_LR}" && $(echo "${ORACLE_LR} > 0" | bc -l) -eq 1 ]]; then
        additionnal_args+=('--oracle_lr' "${ORACLE_LR}")
    fi

    if [[ -n "${DATASET_TO_AUGMENT}" ]]; then
        additionnal_args+=('--dataset_to_augment' "${DATASET_TO_AUGMENT}")
    fi

    if [ $DISABLE_ORACLE_TRAINING -eq 1 ]; then
        echo "DISABLE_ORACLE_TRAINING is set to 1. Disabling oracle training."
        additionnal_args+=('--disable_oracle_training')
    fi

    if [ $ALG == "PPO" ]; then
        additionnal_args+=('--entropy_loss_coeff' "${ENTROPY_LOSS_COEFF}")
        additionnal_args+=('--action_std' "${ACTION_STD}")
        additionnal_args+=('--K_epochs' "${K_EPOCHS}")
        additionnal_args+=('--val_clip_coef' "${VALUECLIP}")
        additionnal_args+=('--eps_clip' "${POLICYCLIP}")
        additionnal_args+=('--kl_penalty_coeff' "${KL_PENALTY_COEFF}")
        additionnal_args+=('--kl_target' "${KL_TARGET}")
        additionnal_args+=('--kl_horizon' "${KL_HORIZON}")

        if [ $ADAPTIVE_KL -eq 1 ]; then
            additionnal_args+=('--adaptive_kl')
        fi

        if [ $INIT_CRITIC_TO_ORACLE -eq 1 ]; then
            additionnal_args+=('--init_critic_to_oracle')
        fi
    fi

    # Start training
    ${PYTHONEXEC} -O $SOURCEDIR/TrackToLearn/trainers/rlhf_refactored_train.py \
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
        --oracle_crit_checkpoint ${ORACLE_CRIT_CHECKPOINT} \
        --oracle_reward_checkpoint ${ORACLE_REWARD_CHECKPOINT} \
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
        --alg ${ALG} \
        --oracle_batch_size ${ORACLE_MICRO_BATCH_SIZE} \
        --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
        "${additionnal_args[@]}"

    # POST-PROCESSING
    bash scripts/tractogram_post_processing.sh ${DEST_FOLDER} ${DATASETDIR}
done

if [ $islocal -eq 1 ]; then
    echo "Done."
    echo "Experiment results are saved in ${EXPDIR}."
else
    # Archive and save everything
    OUTNAME=${EXPID}$(date -d "today" +"%Y%m%d%H%M").tar

    echo "Archiving experiment..."
    tar -cvf ${DATADIR}/${OUTNAME} $EXPDIR
    echo "Copying archive to scratch..."
    cp ${DATADIR}/${OUTNAME} ~/scratch/${OUTNAME}

    echo "Done."
    echo "Experiment results are saved in ~/scratch/${OUTNAME}."
fi
