from abc import abstractmethod, ABCMeta
from dipy.io.stateful_tractogram import StatefulTractogram

class Filterer(metaclass=ABCMeta):
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, reference, tractograms, gt_config, out_dir, scored_extension="trk", tmp_base_dir=None):
        """Filter a list of tracts."""
        raise NotImplementedError
    
    