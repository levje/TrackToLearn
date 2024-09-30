import os
import h5py
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram
from tqdm import tqdm

DEFAULT_DATASET_NAME = "new_dataset.hdf5"
"""
The StreamlineDatasetManager manages a HDF5 file containing the streamlines
and their respective scores used to train the Oracle model. Here we manage
the creation of the dataset if needed, and the addition of new streamlines
to the HDF5 file (the dataset).

The latter has a specific structure:
attrs:
    - version: the version of the dataset
    - nb_points: the number of points in the streamlines (e.g. 128)

dataset:
    - train (~90% of the data)
        - streamlines (N, 128, 3): the streamlines
        - scores (N,): the class/scores of the streamlines (0 or 1)
    - test (~10% of the data)
        - streamlines (N, 128, 3): the streamlines
        - scores (N,): the class/scores of the streamlines (0 or 1)
"""


class StreamlineDatasetManager(object):
    def __init__(self,
                 saving_path: str,
                 dataset_to_augment_path: str = None,
                 augment_in_place: bool = False,
                 dataset_name: str = DEFAULT_DATASET_NAME,
                 number_of_points: int = 128,
                 test_ratio: float = 0.1,
                 add_batch_size: int = 1000):

        assert 0 <= test_ratio <= 1, "The test ratio must be between 0 and 1."

        self.test_ratio = test_ratio
        self.train_ratio = 1 - test_ratio
        self.number_of_points = number_of_points
        self.add_batch_size = add_batch_size

        if not os.path.exists(saving_path) and not saving_path == "":
            raise FileExistsError(
                f"The saving path {saving_path} does not exist.")

        if dataset_to_augment_path is not None:
            print("Loading the dataset to augment: ", dataset_to_augment_path)
            self.current_train_nb_streamlines, self.current_test_nb_streamlines = self._load_and_verify_streamline_dataset(
                dataset_to_augment_path)

            if not augment_in_place:
                # We don't want to modify the original dataset, so we copy it to the saving_path
                self.dataset_file_path = os.path.join(
                    saving_path, dataset_name)

                # Copy the dataset to the saving path
                with h5py.File(dataset_to_augment_path, 'r') as original:
                    with h5py.File(self.dataset_file_path, 'w') as target:
                        original.copy('train', target)
                        original.copy('test', target)
                        target.attrs['version'] = original.attrs['version']
                        target.attrs['nb_points'] = original.attrs['nb_points']
            else:
                # Let's just use the original dataset file.
                self.dataset_file_path = dataset_to_augment_path

            self.file_is_created = True
        else:
            print("Creating a new dataset.")
            self.dataset_file_path = os.path.join(saving_path, dataset_name)
            self.current_train_nb_streamlines = 0
            self.current_test_nb_streamlines = 0
            self.file_is_created = False

    def add_tractograms_to_dataset(self, filtered_tractograms: list[tuple[StatefulTractogram,
                                                                          StatefulTractogram]]):
        """ Gathers all the filtered tractograms and creates or appends a dataset for the
        reward model training. Outputs into a hdf5 file."""

        if len(filtered_tractograms) == 0:
            import warnings
            warnings.warn(
                "Called add_tractograms_to_dataset with an empty list of tractograms.")

        # For each sft, get the indices of streamlines for training or for testing.
        # We do that before adding anything to the dataset, because we want to know
        # the total number of streamlines to add for each set (train/test) to resize
        # the dataset accordingly.
        train_indices = []          # [(pos_indices, neg_indices), ...]
        test_indices = []           # [(pos_indices, neg_indices), ...]
        train_nb_streamlines = 0    # train_nb_pos + train_nb_neg
        test_nb_streamlines = 0     # test_nb_pos + test_nb_neg
        for sft_valid, sft_invalid in filtered_tractograms:
            # Positive
            nb_pos = len(sft_valid.streamlines)
            nb_pos_train = int(nb_pos * self.train_ratio)
            nb_pos_test = nb_pos - nb_pos_train

            pos_indices = np.random.choice(nb_pos, nb_pos, replace=False)
            pos_train_indices = pos_indices[:nb_pos_train]
            pos_test_indices = pos_indices[nb_pos_train:]

            # Negative
            nb_neg = len(sft_invalid.streamlines)
            nb_neg_train = int(nb_neg * self.train_ratio)
            nb_neg_test = nb_neg - nb_neg_train

            neg_indices = np.random.choice(nb_neg, nb_neg, replace=False)
            neg_train_indices = neg_indices[:nb_neg_train]
            neg_test_indices = neg_indices[nb_neg_train:]

            # Add to the list of indices
            train_indices.append((pos_train_indices, neg_train_indices))
            test_indices.append((pos_test_indices, neg_test_indices))

            train_nb_streamlines += nb_pos_train + nb_neg_train
            test_nb_streamlines += nb_pos_test + nb_neg_test

        write_mode = 'w' if not self.file_is_created else 'a'
        with h5py.File(self.dataset_file_path, write_mode) as f:

            # Create the hdf5 file structure if not already done
            if not self.file_is_created:
                f.attrs['version'] = 1
                f.attrs['nb_points'] = self.number_of_points
                direction_dimension = \
                    filtered_tractograms[0][0].streamlines[0].shape[-1] \
                    if len(filtered_tractograms[0][0].streamlines) > 0 \
                    else filtered_tractograms[0][1].streamlines[0].shape[-1]

                # Create the train/test groups
                train_group = f.create_group('train')
                test_group = f.create_group('test')

                # Create the TRAIN dataset (train/data & train/scores)
                train_group.create_dataset(
                    'data',
                    shape=(train_nb_streamlines,
                           self.number_of_points, direction_dimension),
                    maxshape=(None, self.number_of_points, direction_dimension))

                train_group.create_dataset(
                    'scores',
                    shape=(train_nb_streamlines,),
                    maxshape=(None,))

                # Create the TEST dataset (test/data & test/scores)
                test_group.create_dataset(
                    'data',
                    shape=(test_nb_streamlines, self.number_of_points,
                           direction_dimension),
                    maxshape=(None, self.number_of_points, direction_dimension))

                test_group.create_dataset(
                    'scores',
                    shape=(test_nb_streamlines,),
                    maxshape=(None,))

                self.file_is_created = True
                do_resize = False

            # The dataset file is already created. Make sure it's
            # consistent with the current dataset.
            else:
                assert f.attrs['nb_points'] == self.number_of_points, \
                    "The number of points in the dataset is different from the one in the manager."
                train_group = f['train']
                test_group = f['test']
                f.attrs['version'] += 1
                # We are appending new data, we need to resize the dataset.
                do_resize = True

            # Resize the dataset to append the new streamlines.
            if do_resize:
                train_group['data'].resize(
                    self.current_train_nb_streamlines + train_nb_streamlines, axis=0)
                train_group['scores'].resize(
                    self.current_train_nb_streamlines + train_nb_streamlines, axis=0)

                test_group['data'].resize(
                    self.current_test_nb_streamlines + test_nb_streamlines, axis=0)
                test_group['scores'].resize(
                    self.current_test_nb_streamlines + test_nb_streamlines, axis=0)

            # Indices where to add the streamlines in the file (contiguous at the end of array).
            file_train_indices = np.arange(
                self.current_train_nb_streamlines, self.current_train_nb_streamlines + train_nb_streamlines)
            file_test_indices = np.arange(
                self.current_test_nb_streamlines, self.current_test_nb_streamlines + test_nb_streamlines)

            np.random.shuffle(file_train_indices)
            np.random.shuffle(file_test_indices)

            # Actually add the streamlines to the dataset using the precalculated
            # indices.
            for i, (sft_train_indices, sft_test_indices) in enumerate(tqdm(zip(train_indices, test_indices), desc="Adding tractograms to dataset", total=len(filtered_tractograms))):
                # Unpack and setup
                pos_train_indices, neg_train_indices = sft_train_indices
                pos_test_indices, neg_test_indices = sft_test_indices

                valid_sft, invalid_sft = filtered_tractograms[i]
                valid_sft.to_vox()
                valid_sft.to_corner()
                invalid_sft.to_vox()
                invalid_sft.to_corner()

                # Add the training positive streamlines
                file_idx = file_train_indices[:len(pos_train_indices)]
                self._add_streamlines_to_hdf5(train_group,
                                              valid_sft[pos_train_indices],
                                              self.number_of_points,
                                              file_idx,
                                              sub_pbar_desc="add train/pos streamlines",
                                              batch_size=self.add_batch_size)
                file_train_indices = file_train_indices[len(
                    pos_train_indices):]

                # Add the training negative streamlines
                file_idx = file_train_indices[:len(neg_train_indices)]
                self._add_streamlines_to_hdf5(train_group,
                                              invalid_sft[neg_train_indices],
                                              self.number_of_points,
                                              file_idx,
                                              sub_pbar_desc="add train/neg streamlines",
                                              batch_size=self.add_batch_size)
                file_train_indices = file_train_indices[len(
                    neg_train_indices):]

                # Add the testing positive streamlines
                file_idx = file_test_indices[:len(pos_test_indices)]
                self._add_streamlines_to_hdf5(test_group,
                                              valid_sft[pos_test_indices],
                                              self.number_of_points,
                                              file_idx,
                                              sub_pbar_desc="add test/pos streamlines",
                                              batch_size=self.add_batch_size)
                file_test_indices = file_test_indices[len(pos_test_indices):]

                # Add the testing negative streamlines
                file_idx = file_test_indices[:len(neg_test_indices)]
                self._add_streamlines_to_hdf5(test_group,
                                              invalid_sft[neg_test_indices],
                                              self.number_of_points,
                                              file_idx,
                                              sub_pbar_desc="add test/neg streamlines",
                                              batch_size=self.add_batch_size)
                file_test_indices = file_test_indices[len(neg_test_indices):]

            assert len(
                file_train_indices) == 0, "Not all training streamlines were added."
            assert len(
                file_test_indices) == 0, "Not all testing streamlines were added."

            self.current_train_nb_streamlines += train_nb_streamlines
            self.current_test_nb_streamlines += test_nb_streamlines

    def _add_streamlines_to_hdf5(self, f, sft, nb_points, idx, sub_pbar_desc="", batch_size=1000):
        """ Add the streamlines to the hdf5 file.

        Parameters
        ----------
        hdf_subject: h5py.File
            HDF5 file to save the dataset to.
        sft: nib.streamlines.tractogram.Tractogram
            Streamlines to add to the dataset.
        nb_points: int, optional
            Number of points to resample the streamlines to
        total: int
            Total number of streamlines in the dataset
        idx: list
            List of positions to store the streamlines
        """

        # Get the scores and the streamlines
        scores = np.asarray(
            sft.data_per_streamline['score'], dtype=np.uint8).squeeze(-1)
        # Resample the streamlines
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        streamlines = np.asarray(streamlines)
        idx = np.sort(idx)

        data_group = f['data']
        scores_group = f['scores']

        num_batches = (len(idx) // batch_size) + (len(idx) % batch_size != 0)

        for batch_start in tqdm(range(0, len(idx), batch_size), desc=sub_pbar_desc, total=num_batches, leave=False):
            batch_end = min(batch_start + batch_size, len(idx))
            batch_idx = idx[batch_start:batch_end]
            batch_streamlines = streamlines[batch_start:batch_end]
            batch_scores = scores[batch_start:batch_end]

            data_group[batch_idx] = batch_streamlines
            scores_group[batch_idx] = batch_scores

    def _load_and_verify_streamline_dataset(self, dataset_to_augment_path: str):
        """ Verify the dataset in the hdf5 file."""
        def get_group_size(group):
            if 'data' not in group:
                raise ValueError(
                    f"The dataset ({group}) does not contain the 'data' group.")

            if 'scores' not in group:
                raise ValueError(
                    f"The dataset ({group}) does not contain the 'scores' group.")

            return group['scores'].shape[0]

        with h5py.File(dataset_to_augment_path, 'r') as dataset:

            has_train = 'train' in dataset
            has_test = 'test' in dataset

            if not has_train and not has_test:
                raise ValueError(
                    "The dataset does not contain the 'train' or 'test' groups.")

            train_size = get_group_size(dataset['train']) if has_train else 0
            test_size = get_group_size(dataset['test']) if has_test else 0

            return train_size, test_size
