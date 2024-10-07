EXPNAME=OraclePaperPredict-AntoineDS
EXPPATH=data/experiments/TractOracleNet/${EXPNAME}
EXPID=Training1
MAXEPOCHS=50

DATASET_FILE=$1
if [ -z "$DATASET_FILE" ]; then
    # Antoine's classic dataset
    DATASET_FILE=/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine.hdf5

    # Jeremi's classic dataset
    # DATASET_FILE=/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/train_test_classical_tracts_dataset.hdf5

    # Jeremi's SAC dataset
    # DATASET_FILE="/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/streamlines/stable/train_test_sac_even_bigger_dataset_eq.hdf5"
fi

# Make sure the dataset file exists and is not empty
if [ ! -f "${DATASET_FILE}" ]; then
    echo "Dataset file not found: ${DATASET_FILE}"
    exit 1
fi

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
echo "Using dataset: ${DATASET_FILE}"

python TrackToLearn/runners/tractoracle_predict.py \
    ${EXPPATH} \
    ${EXPNAME} \
    ${EXPID} \
    ${MAXEPOCHS} \
    ${DATASET_FILE} \
    --oracle_batch_size ${MICRO_BATCH_SIZE} \
    --oracle_checkpoint "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/experiments/TractOracleNet/OracleTrainTest/OracleTrainTest/Training-DenseFalse-/best_vc_epoch.ckpt"
    # --oracle_checkpoint "custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt"


