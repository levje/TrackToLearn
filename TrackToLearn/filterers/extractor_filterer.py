from TrackToLearn.filterers.filterer import Filterer
from dipy.io.stateful_tractogram import StatefulTractogram

import argparse
import tempfile
import numpy as np
import subprocess

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path
from typing import Union

class ExtractorFilterer(Filterer):
        
    def __init__(self):
        super(ExtractorFilterer, self).__init__()

    def __call__(self, reference, tractograms, gt_config, out_dir, scored_extension="trk", tmp_base_dir=None):
        filtered_tractograms = []

        for tractogram in tractograms:
                tractogram_path = Path(tractogram)
                assert tractogram_path.exists(), f"Tractogram {tractogram} does not exist."
    
                with tempfile.TemporaryDirectory(dir=tmp_base_dir) as tmp:
    
                    tmp_path = Path(tmp)
    
                    out_path = out_dir / "scored_{}.{}".format(tractogram_path.stem, scored_extension)
                    self._merge_with_scores(reference, tractogram_path, gt_config, out_path)
                    
                    filtered_tractograms.append(out_path)

        return filtered_tractograms

