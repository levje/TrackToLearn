import os
import tempfile
import numpy as np
import time
from collections import namedtuple

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from scilpy.segment.tractogram_from_roi import _extract_vb_and_wpc_all_bundles

from TrackToLearn.experiment.tractometer_validator import load_and_verify_everything
from TrackToLearn.environments.reward import Reward
import cProfile, pstats, io
from pstats import SortKey

class TractometerReward(Reward):

    """ Reward streamlines based on the predicted scores of an "Oracle".
    A binary reward is given by the oracle at the end of tracking.
    """

    def __init__(
        self,
        base_dir,
        reference,
        min_nb_steps: int,
        affine_vox2rasmm: np.ndarray
    ):
        print("Initializing TractometerReward...")
        # Name for stats
        self.name = 'tractometer_reward'
        # Minimum number of steps before giving reward
        # Only useful for 'sparse' reward
        self.min_nb_steps = min_nb_steps

        self.reference = reference
        self.affine_vox2rasmm = affine_vox2rasmm
        self.gt_config = os.path.join(base_dir, 'scil_scoring_config.json')
        self.gt_dir = base_dir
        self.dilation_factor = 1

        self.temp = tempfile.mkdtemp()
        args_mocker = namedtuple('args', [
            'compute_ic', 'save_wpc_separately', 'unique', 'reference',
            'bbox_check', 'out_dir', 'dilate_endpoints', 'no_empty'])
        self.args = args_mocker(
                False, False, True, self.reference, False, self.temp,
                self.dilation_factor, False)
        

        # Load
        (self.gt_tails, self.gt_heads, self.bundle_names, self.list_rois,
         self.bundle_lengths, self.angles, self.orientation_lengths,
         self.abs_orientation_lengths, self.inv_all_masks, self.gt_masks,
         self.any_masks) = \
            load_and_verify_everything(
                self.reference,
                self.gt_config,
                self.gt_dir,
                False)

    def __del__(self):
        os.rmdir(self.temp)

    def reward(self, sft: StatefulTractogram, dones):

        reward = np.zeros((dones.shape[0]))

        _, _, detected_vs_wpc_ids, _ = \
            _extract_vb_and_wpc_all_bundles(
                self.gt_tails, self.gt_heads, sft, self.bundle_names, self.bundle_lengths,
                self.angles, self.orientation_lengths, self.abs_orientation_lengths,
                self.inv_all_masks, self.any_masks, self.args)

        idx = np.arange(dones.shape[0])[dones][detected_vs_wpc_ids.astype(int)] # NB: .astype(int) is needed whenever the list is empty.
        reward[idx] = 1.0
        return reward


    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):

        N, L, _ = streamlines.shape
        if L > self.min_nb_steps and sum(dones.astype(int)) > 0:

            # Change ref of streamlines. This is weird on the ISMRM2015
            # dataset as the diff and anat are not in the same space,
            # but it should be fine on other datasets.
            tractogram = Tractogram(
                streamlines=streamlines.copy()[dones])
            tractogram.apply_affine(self.affine_vox2rasmm)
            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)
            sft.to_vox()
            sft.to_corner()
            return self.reward(sft, dones)
        return np.zeros((N))
