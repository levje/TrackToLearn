import comet_ml

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
            warnings.warn("Comet is not being used. No metrics will be logged for the Oracle training.")

    def log_parameters(self, hyperparameters: dict):
        self.e.log_parameters(hyperparameters)

    def log_metrics(self, metrics_dict: dict, episode: int):

        if not self.use_comet:
            return
        elif episode < 0 or not isinstance(episode, int):
            raise ValueError("Episode number must be a positive integer, not {}".format(episode))
        elif len(metrics_dict) == 0:
            return

        prefix = None if self.prefix == '' else self.prefix
        self.experiment.log_metrics(metrics_dict, step=episode, prefix=prefix)
        