
NB_STREAMLINES=$1
ALGORITHM=$2
OUTNAME=$3
DATASET_DIR=$4

# The script requires at least 3 arguments. The 4th argument is optional and will
# have a default value if not provided.

if [ "$#" -lt 3 ]; then
	echo "Usage: track_mrtrix.sh <nb_streamlines> <algorithm> <outname> [dataset_dir]"
	exit 1
fi

# Check if the dataset directory is provided
if [ "$#" -eq 3 ]; then
	DATASET_DIR="data/datasets/fibercup/"
fi



# Check the algorithm; it should only be [FACT, iFOD1, iFOD2, SD_Stream]
if [ "$ALGORITHM" != "FACT" ] && [ "$ALGORITHM" != "iFOD1" ] && [ "$ALGORITHM" != "iFOD2" ] && [ "$ALGORITHM" != "SD_Stream" ]; then
	echo "Invalid algorithm"
	echo "Valid algorithms: [FACT, iFOD1, iFOD2, SD_Stream]"
	echo "Usage: track_mrtrix.sh <nb_streamlines> <algorithm> <outname>"
	exit 1
fi

MINLENGTH=3
MAXLENGTH=200

# Track the interface using iFOD2
tckgen ${DATASET_DIR}/fodfs/fibercup_fodf_mrtrix.nii.gz \
	${OUTNAME} \
	-algorithm ${ALGORITHM} \
	-select ${NB_STREAMLINES} \
	-minlength ${MINLENGTH} \
	-maxlength ${MAXLENGTH} \
	-seed_image ${DATASET_DIR}/maps/interface.nii.gz
