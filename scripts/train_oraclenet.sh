islocal=1

# if is local
if [ $islocal -eq 1 ]; then
    echo "Running locally"
else
    echo "Running on HPC..."
fi

EXPNAME=OracleTrainTestFibercup
EXPPATH=data/experiments/TractOracleNet/${EXPNAME}
EXPID=Transformer-Classif-Zero-Big-MSELoss-
MAXEPOCHS=500
#DATASET_FILE=/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/train_test_classical_tracts_dataset.hdf5
# DATASET_FILE=antoine-pft.hdf5
# DATASET_FILE=full-antoine.hdf5
# DATASET_FILE=data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine_modrange.hdf5
# DATASET_FILE=data/datasets/fibercup/streamlines/stable/fibercup_tracts.hdf5
DATASET_FILE=data/datasets/fibercup/streamlines/stable/fibercup_tracts_big.hdf5

mkdir -p ${EXPPATH}

# BATCH SIZE and GRAD ACCUMULATION
# The original batch size is 2816, but since we want to use a significantly smaller
# batch size, we need to increase the number of gradient accumulation steps to
# compensate for the smaller batch size. The original batch size is 2048 and we want
# to use a batch size of 512.
TOTAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=2048 #512 # Should reduce or increase this based on the GPU memory available.
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / MICRO_BATCH_SIZE)) # 88

echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"

python TrackToLearn/trainers/tractoraclenet_train.py \
    ${EXPPATH} \
    ${EXPNAME} \
    ${EXPID} \
    ${MAXEPOCHS} \
    ${DATASET_FILE} \
    --lr 0.0001 \
    --oracle_batch_size ${MICRO_BATCH_SIZE} \
    --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --use_comet \
    --n_head 4 \
    --n_layers 4 \
    --out_activation sigmoid 
    # --dense \
    # --partial


