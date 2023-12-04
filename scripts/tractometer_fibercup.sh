#!/bin/sh

if [ $# -eq 0 ] || [ $1 = '-h' ] || [ $1 = '--help' ]
then
    echo "-----"
    echo "Usage: "
    echo ">> scil_score_ismrm_Renauld2023.sh tractogram out_dir scoring_data"
    echo "-----"
    exit
fi

tractogram=$1
out_dir=$2
scoring_data=$3

config_file_segmentation=$scoring_data/scoring_config.json
config_file_tractometry=$scoring_data/scil_scoring_config.json
scores_ref=fibercup_scores_ref.json

if [ ! -f $tractogram ]
then
    echo "Tractogram $tractogram does not exist"
    exit
fi

if [ -d $out_dir ]
then
    echo "Out dir $out_dir already exists. Delete first."
    exit 1
fi

echo '------------- TRACTOMETRY ------------'
scil_score_tractogram.py $tractogram $config_file_tractometry $out_dir --no_empty \
    --reference $scoring_data/../dti/fibercup_fa.nii.gz --gt_dir $scoring_data --no_bbox_check --unique --compute_ic -v --dilate 3;

echo '----------------  DONE ---------------'

cat $out_dir/results.json

echo '------ COMPARE WITH REFERENCE --------'
if [ -f $scores_ref ]
then
    python compare_scores_with_reference.py --reference $scores_ref --scores $out_dir/results.json
else
    echo "Skipping scores comparison. Can't compare with reference if $scores_ref does not exist"
fi

echo '----------------  DONE ---------------'
