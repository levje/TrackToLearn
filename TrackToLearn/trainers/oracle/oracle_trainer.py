import torch
from TrackToLearn.oracles.transformer_oracle import TransformerOracle
from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.utils.torch_utils import get_device

class OracleTrainer:
    def __init__(self, checkpoint, batch_size=4096) -> None:
        # This should train the oracle used in the reward & stopping criterion.
        self.model = OracleSingleton(checkpoint, get_device(), batch_size=batch_size) 

    def train(self, model: TransformerOracle, batch: torch.Tensor) -> torch.Tensor:
        """
        Train the model for n steps.
        """
        return model(batch)
