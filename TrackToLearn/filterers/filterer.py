from abc import abstractmethod, ABCMeta

class TractsFilterer(metaclass=ABCMeta):
    
    def __init__(self):
        pass

    @abstractmethod
    def filter(self, tractogram):
        """Filter a list of tracts."""
        raise NotImplementedError
    