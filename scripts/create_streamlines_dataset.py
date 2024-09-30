import argparse
import h5py
import json
import os
from glob import glob
from os.path import expanduser
from dipy.io.streamline import load_tractogram
from TrackToLearn.trainers.oracle.streamline_dataset_manager import StreamlineDatasetManager
import numpy as np


def generate_dataset(
    config_file: str,
    dataset_file: str
) -> None:
    """ Generate a dataset from a configuration file and save it to disk.

    Parameters:
    -----------
    config_file: str
        Path to the configuration file containing the subjects and their
        streamlines.
    dataset_file: str
        Path to the output file where the dataset will be saved.
    nb_points: int
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """
    # Get the dataset_file folder
    dataset_folder = os.path.dirname(dataset_file)
    dataset_name = os.path.basename(dataset_file)
    dataset_dir = os.path.abspath(os.path.join(
        os.path.dirname(config_file), os.pardir))

    dataset_manager = StreamlineDatasetManager(
        saving_path=dataset_folder, dataset_name=dataset_name, add_batch_size=1000)
    with open(config_file, "r") as conf:
        config = json.load(conf)
        # Config:
        # {
        #     "ismrm2015_1mm": {
        #         "streamlines": [
        #             "ismrm2015_1mm/tractograms/scored_tractograms/*"
        #         ],
        #         "reference": "ismrm2015/anat/ismrm2015_T1.nii.gz"
        #     }
        # }

        for subject_id in config:  # subject_id = ismrm2015_1mm
            subject = config[subject_id]
            streamlines_files_list = subject["streamlines"]

            streamlines_files_exp = add_path_prefix_if_needed(
                dataset_dir, streamlines_files_list[0])
            expanded = expanduser(streamlines_files_exp)
            streamlines_files = glob(expanded)
            reference_anat = add_path_prefix_if_needed(
                dataset_dir, subject["reference"])

            tractograms = []
            for streamlines_file in streamlines_files:
                print("Loading streamlines from file : {}".format(streamlines_file))
                valid_streamlines, invalid_streamlines = load_streamlines(
                    streamlines_file, reference_anat)
                tractograms.append((valid_streamlines, invalid_streamlines))

            assert tractograms, "No streamlines were loaded for subject {}".format(
                subject_id)
            dataset_manager.add_tractograms_to_dataset(tractograms)

    print("Saved dataset : {}".format(dataset_file))


def load_streamlines(
    streamlines_file: str,
    reference,
):
    """ Load the streamlines from a file and make sure they are in
    voxel space and aligned with the corner of the voxels.

    Parameters
    ----------
    streamlines_file: str
        Path to the file containing the streamlines.
    reference: str
        Path to the reference anatomy file.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    """

    sft = load_tractogram(streamlines_file, reference, bbox_valid_check=False)
    sft.to_corner()
    sft.to_vox()

    # TODO: This won't work, we need to get the streamlines data probably iteratively.
    scores = sft.data_per_streamline["score"].squeeze(1)
    indices = np.arange(len(scores))
    valid_indices = indices[scores == 1]
    invalid_indices = indices[scores == 0]

    return sft[valid_indices], sft[invalid_indices]


def add_path_prefix_if_needed(path_prefix: str, file: str) -> str:
    """ Add the path to the configuration file if it is not an absolute path.

    Parameters:
    -----------
    config_file: str
        Path to the configuration file.

    Returns:
    --------
    str
        The absolute path to the configuration file.
    """
    if not os.path.isabs(file):
        return os.path.abspath(os.path.join(path_prefix, file))
    return file


def parse_args():

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__)

    parser.add_argument('config_file', type=str,
                        help="Configuration file to load subjects and their"
                        " volumes.")
    parser.add_argument('output', type=str,
                        help="Output filename including path.")

    arguments = parser.parse_args()

    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    generate_dataset(config_file=args.config_file,
                     dataset_file=args.output)


if __name__ == "__main__":
    main()
