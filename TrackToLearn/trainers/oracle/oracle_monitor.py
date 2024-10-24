import comet_ml
import logging
import numpy as np


class OracleMonitor(object):

    def __init__(
        self,
        experiment: comet_ml.Experiment,
        experiment_id: str,
        use_comet: bool = False,
        metrics_prefix: str = None
    ):
        self.experiment_id = experiment_id
        self.experiment = experiment

        self.experiment.add_tag(experiment_id)
        self.experiment.set_name(experiment_id)

        delims = ['_', '-', '/']
        if metrics_prefix[0] in delims:
            self.metrics_prefix = metrics_prefix[1:]
        else:
            self.metrics_prefix = f"{metrics_prefix}/"

        self.use_comet = use_comet
        if not self.use_comet:
            import warnings
            warnings.warn(
                "Comet is not being used. No metrics will be logged for the Oracle training.")
            
        self.metrics_prefix = metrics_prefix

    def log_parameters(self, hyperparameters: dict):
        if not self.use_comet:
            return
        
        if self.metrics_prefix:
                prefix = f"{self.metrics_prefix}/"
        
        self.experiment.log_parameters(hyperparameters, prefix=self.metrics_prefix)

    def log_metrics(self, metrics_dict, step: int, epoch: int):
        if not self.use_comet:
            return

        for k, v in metrics_dict.items():
            assert isinstance(v, (int, float, np.int64, np.float64,
                              np.float32, np.int32)), "Metrics must be numerical."
            
            if self.metrics_prefix:
                k = f"{self.metrics_prefix}/{k}"

            self.experiment.log_metric(k, v, step=step)
