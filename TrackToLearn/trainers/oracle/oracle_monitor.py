import comet_ml
import logging
import numpy as np


class OracleMonitor(object):

    def __init__(
        self,
        experiment: comet_ml.Experiment,
        experiment_id: str,
        prefix: str = '',
        use_comet: bool = False
    ):
        self.experiment_id = experiment_id
        self.experiment = experiment

        self.experiment.add_tag(experiment_id)
        self.experiment.set_name(experiment_id)

        self.prefix = prefix
        self.use_comet = use_comet
        if not self.use_comet:
            import warnings
            warnings.warn(
                "Comet is not being used. No metrics will be logged for the Oracle training.")

    def log_parameters(self, hyperparameters: dict):
        if not self.use_comet:
            return
        self.experiment.log_parameters(hyperparameters)

    # def log_metrics(self, metrics_dict: dict, episode: int):

    #     if not self.use_comet:
    #         return
    #     elif episode < 0 or not isinstance(episode, int):
    #         raise ValueError("Episode number must be a positive integer, not {}".format(episode))
    #     elif len(metrics_dict) == 0:
    #         logging.warning("No metrics to log.")
    #         print("No metrics to log.")
    #         return
    #     print("Should log normally to comet")
    #     prefix = None if self.prefix == '' else self.prefix
    #     self.experiment.log_metrics(metrics_dict, step=episode, prefix=prefix)

    def log_metrics(self, metrics_dict, step: int, epoch: int):
        if not self.use_comet:
            return

        # print("Logging metrics to comet with step {} and epoch {}".format(step, epoch))
        for k, v in metrics_dict.items():
            assert isinstance(v, (int, float, np.int64, np.float64,
                              np.float32, np.int32)), "Metrics must be numerical."
            self.experiment.log_metric(k, v, step=step)
