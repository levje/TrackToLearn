#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-10:00:00

module load python/3.10 cuda cudnn

SOURCEDIR=~/TrackToLearn
DATADIR=$SLURM_TMPDIR/data
EXPDIR=$SLURM_TMPDIR/experiments
LOGSDIR=$SLURM_TMPDIR/logs

# Prepare virtualenv
echo "Sourcing ENV-TTL-2 virtual environment..."
source ~/ENV-TTL-2/bin/activate # Ideally, we would build the environnement within the node itself, but too much dependencies for now.

# Prepare datasets
mkdir $DATADIR
mkdir $EXPDIR

echo "Unpacking datasets..."
tar xf ~/projects/def-pmjodoin/levj1404/datasets/ismrm2015/ismrm_ttl_dataset.tar -C $DATADIR

echo "Copying oracle checkpoint..."
cp ~/projects/def-pmjodoin/levj1404/oracles/ismrm_paper_oracle.ckpt $DATADIR

# Parameters
EXPNAME="ISMRM-TractOracleRL-OraclePaper"
EXPID="OraclePaper"
MAXEP=10
BATCHSIZE=4096
SEED=1111
ORACLECHECKPOINT=$DATADIR/ismrm_paper_oracle.ckpt
NPV=4

# Start training
python $SOURCEDIR/TrackToLearn/trainers/sac_auto_train.py \
    "$EXPDIR/$EXPNAME" \
    "$EXPNAME" \
    "$EXPID" \
    "$DATADIR/ismrm2015.hdf5" \
    --max_ep $MAXEP \
    --oracle_checkpoint $ORACLECHECKPOINT \
    --oracle_validator \
    --oracle_stopping_criterion \
    --oracle_bonus 10.0 \
    --scoring_data "$DATADIR/scoring_data" \
    --tractometer_reference "$DATADIR/scoring_data/" \
    --tractometer_validator \
    --use_comet \
    --comet_offline_dir $LOGSDIR \
    --workspace "mrzarfir" \
    --rng_seed $SEED \
    --n_actor 4096 \
    --npv ${NPV} \
    --min_length 20 \
    --max_length 200 \
    --noise 0.0 \
    --batch_size $BATCHSIZE \
    --replay_size 1000000

# Archive and save everything
OUTNAME=${EXPNAME}$(date +"%F").tar

echo "Archiving experiment..."
tar -cvf ${DATADIR}/${OUTNAME} $EXPDIR $LOGSDIR
echo "Copying archive to scratch..."
cp ${DATADIR}/${OUTNAME} ~/scratch/${OUTNAME}