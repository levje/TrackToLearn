import os
import tempfile
from collections import namedtuple

import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.segment.tractogram_from_roi import segment_tractogram_from_roi
from scilpy.tractanalysis.scoring import compute_tractometry

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram

from TrackToLearn.filterers.filterer import Filterer
from TrackToLearn.experiment.tractometer_validator import load_and_verify_everything
from pathlib import Path

class TractometerFilterer(Filterer):

    def __init__(
        self,
        base_dir,
        reference,
        dilate_endpoints=1,
    ):
        self.name = 'Tractometer'
        self.gt_config = os.path.join(base_dir, 'scil_scoring_config.json')
        self.gt_dir = base_dir
        self.dilation_factor = dilate_endpoints

        assert os.path.exists(reference), f"Reference {reference} does not exist."
        self.reference = reference

        # Load
        (self.gt_tails, self.gt_heads, self.bundle_names, self.list_rois,
         self.bundle_lengths, self.angles, self.orientation_lengths,
         self.abs_orientation_lengths, self.inv_all_masks, self.gt_masks,
         self.any_masks) = \
            load_and_verify_everything(
                reference,
                self.gt_config,
                self.gt_dir,
                False)

    def __call__(self, tractogram, out_dir, scored_extension="trk"):
        assert os.path.exists(tractogram), f"Tractogram {tractogram} does not exist."
        filtered_path = os.path.join(out_dir, "scored_{}.{}".format(Path(tractogram).stem, scored_extension))
        sft = load_tractogram(tractogram, self.reference,
                            bbox_valid_check=True, trk_header_check=True)
        
        if len(sft.streamlines) == 0:
            save_tractogram(sft, filtered_path)
            return filtered_path

        args_mocker = namedtuple('args', [
            'compute_ic', 'save_wpc_separately', 'unique', 'reference',
            'bbox_check', 'out_dir', 'dilate_endpoints', 'no_empty'])

        with tempfile.TemporaryDirectory() as temp:
            
            args = args_mocker(
                False, False, True, self.reference, False, temp,
                self.dilation_factor, False)

            # Segment VB, WPC, IB
            (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
            ib_names, _) = segment_tractogram_from_roi(
                sft, self.gt_tails, self.gt_heads, self.bundle_names,
                self.bundle_lengths, self.angles, self.orientation_lengths,
                self.abs_orientation_lengths, self.inv_all_masks, self.any_masks,
                self.list_rois, args)
            
            scored_tractogram = self._merge_with_scores(vb_sft_list, nc_sft, filtered_path)
            save_tractogram(scored_tractogram, filtered_path) # Replace saving with directly putting that data into a hdf5 file.

        return scored_tractogram

    def _merge_with_scores(self, vb_sft_list, inv_tractogram, output):
        """Merge the streamlines with the scores."""
        main_tractogram = None

        # Add valid streamlines
        for bundle in vb_sft_list:
            num_streamlines = len(bundle.streamlines)

            bundle.data_per_streamline['score'] = np.ones(num_streamlines, dtype=np.float32)

            if main_tractogram is None:
                main_tractogram = bundle
            else:
                main_tractogram = main_tractogram + bundle


        # Add invalid streamlines
        num_streamlines = len(inv_tractogram.streamlines)
        if num_streamlines > 0:
            inv_tractogram.data_per_streamline['score'] = np.zeros(num_streamlines, dtype=np.float32)

        main_tractogram = main_tractogram + inv_tractogram if main_tractogram is not None else inv_tractogram
        return main_tractogram
        
