import os
import h5py
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram

DEFAULT_DATASET_NAME = "new_dataset.hdf5"

class StreamlineDatasetManager(object):
    def __init__(self,
                 saving_path: str,
                 dataset_to_augment_path: str = None,
                 augment_in_place: bool = False,
                 dataset_name: str = DEFAULT_DATASET_NAME,
                 number_of_points: int = 128):

        if not os.path.exists(saving_path):
            raise FileExistsError(f"The saving path {saving_path} does not exist.")

        if dataset_to_augment_path is not None:
            print("Loading the dataset to augment: ", dataset_to_augment_path)
            self.current_nb_streamlines = self._load_and_verify_streamline_dataset(dataset_to_augment_path)

            if not augment_in_place:
                # We don't want to modify the original dataset, so we copy it to the saving_path
                self.dataset_file_path = os.path.join(saving_path, dataset_name)

                # Copy the dataset to the saving path
                with h5py.File(dataset_to_augment_path, 'r') as original:
                    with h5py.File(self.dataset_file_path, 'w') as target:
                        original.copy('streamlines', target)
                        target.attrs['version'] = original.attrs['version']
                        target.attrs['nb_points'] = original.attrs['nb_points']
            else:
                # Let's just use the original dataset file.
                self.dataset_file_path = dataset_to_augment_path

            self.file_is_created = True
        else:
            print("Creating a new dataset.")
            self.dataset_file_path = os.path.join(saving_path, dataset_name)
            self.current_nb_streamlines = 0
            self.file_is_created = False

        self.number_of_points = number_of_points
            
    def add_tractograms_to_dataset(self, filtered_tractograms: list[StatefulTractogram]):
        """ Gathers all the filtered tractograms and creates or appends a dataset for the
        reward model training. Outputs into a hdf5 file."""

        if len(filtered_tractograms) == 0:
            import warnings
            warnings.warn("Called add_tractograms_to_dataset with an empty list of tractograms.")

        # Compute the total number of streamlines
        nb_new_streamlines = sum([len(sft.streamlines) for sft in filtered_tractograms])
        assert nb_new_streamlines > 0, "No streamlines to add to the dataset."

        indices = np.arange(self.current_nb_streamlines, self.current_nb_streamlines + nb_new_streamlines)
        np.random.shuffle(indices)

        write_mode = 'w' if not self.file_is_created else 'a'
        with h5py.File(self.dataset_file_path, write_mode) as f:

            if not self.file_is_created:
                f.attrs['version'] = 1
                f.attrs['nb_points'] = self.number_of_points
                # Create the dataset if it does not exist.
                if 'streamlines' not in f:
                    direction_dimension = filtered_tractograms[0].streamlines[0].shape[-1]
                    streamlines_group = f.create_group('streamlines')

                    # 'data' will contain the streamlines
                    streamlines_group.create_dataset(
                        'data',
                        shape=(nb_new_streamlines, self.number_of_points, direction_dimension),
                        maxshape=(None, self.number_of_points, direction_dimension))
                    
                    # 'scores' will contain the labels associated with each streamline.
                    streamlines_group.create_dataset('scores', shape=(nb_new_streamlines), maxshape=(None,))
                self.file_is_created = True
                do_resize = False
            else:
                # Make sure it's consistent with the current dataset.
                assert f.attrs['nb_points'] == self.number_of_points, "The number of points in the dataset is different from the one in the manager."
                f.attrs['version'] += 1
                do_resize = True # We are appending new data

            streamlines_group = f['streamlines']
            data_group = streamlines_group['data']
            scores_group = streamlines_group['scores']

            # Resize the dataset to append the new streamlines.
            if do_resize:
                data_group.resize(self.current_nb_streamlines + nb_new_streamlines, axis=0)
                scores_group.resize(self.current_nb_streamlines + nb_new_streamlines, axis=0)

            for sft in filtered_tractograms:
                sft.to_vox()
                sft.to_corner()
                sft_nb_streamlines = len(sft.streamlines)
                ps_indices = np.random.choice(sft_nb_streamlines, sft_nb_streamlines, replace=False)
                idx = indices[:sft_nb_streamlines]

                self._add_streamlines_to_hdf5(f,
                                              sft[ps_indices],
                                              self.number_of_points,
                                              idx)
                
                indices = indices[sft_nb_streamlines:]
            
            self.current_nb_streamlines += nb_new_streamlines

    def _add_streamlines_to_groups(self, data_group, scores_group, sft, nb_points, idx):
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
        scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)
        # Resample the streamlines
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        streamlines = np.asarray(streamlines)

        for i, st, sc in zip(idx, streamlines, scores):
            data_group[i] = st
            scores_group[i] = sc
    
    def _add_streamlines_to_hdf5(self, f, sft, nb_points, idx):
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
        scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)
        # Resample the streamlines
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        streamlines = np.asarray(streamlines)

        for i, st, sc in zip(idx, streamlines, scores):
            f['streamlines/data'][i] = st
            f['streamlines/scores'][i] = sc

    def _load_and_verify_streamline_dataset(self, dataset_to_augment_path: str):
        """ Verify the dataset in the hdf5 file."""
        with h5py.File(dataset_to_augment_path, 'r') as dataset:
            if 'streamlines' not in dataset:
                raise ValueError("The dataset does not contain the 'streamlines' group.")
            
            streamlines_group = dataset['streamlines']

            if 'data' not in streamlines_group:
                raise ValueError("The dataset does not contain the 'data' group.")
            
            if 'scores' not in streamlines_group:
                raise ValueError("The dataset does not contain the 'scores' group.")
            
            return streamlines_group['data'].shape[0]
        
    # def create_dataset(self,
    #                    filtered_tractograms: list[StatefulTractogram],
    #                    out_file: str):
    #     """ Gathers all the filtered tractograms and creates a dataset for the reward model training. 
    #     Outputs into a hdf5 file."""

    #     # Compute the total number of streamlines
    #     nb_streamlines = sum([len(sft.streamlines) for sft in filtered_tractograms])
    #     nb_points = 128
    #     assert nb_streamlines > 0, "No streamlines to create the dataset."

    #     indices = np.arange(nb_streamlines)
    #     np.random.shuffle(indices)

    #     # Add the streamlines to the dataset
    #     with h5py.File(out_file, 'w') as f:
    #         f.attrs['version'] = 1
    #         f.attrs['nb_points'] = nb_points
            
    #         for sft in filtered_tractograms:
    #             sft.to_corner()
    #             sft.to_vox()
    #             sft_nb_streamlines = len(sft.streamlines)
    #             ps_indices = np.random.choice(sft_nb_streamlines, sft_nb_streamlines, replace=False)
    #             idx = indices[:sft_nb_streamlines]

    #             self._add_streamlines_to_hdf5(f, sft[ps_indices], nb_points, nb_streamlines, idx)

    #             indices = indices[sft_nb_streamlines:]
        
    #     return out_file
