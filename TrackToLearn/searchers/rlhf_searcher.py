#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.trainers.rlhf_train import (
    RlhfTrackToLearnTraining,
    parse_args)
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator
device = get_device()
assert_accelerator()


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
            },
            "oracle_bonus": {
                "type": "discrete",
                "values": [10.0, 50.0]},
            # "alg": { # This can be used in the future to compare between PPO and SAC once PPO is working.
            #     "type": "discrete",
            #     "values": ["SACAuto", "PPO"]
            # }
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
            "seed": args.rng_seed,
            "retryLimit": 3,
            "retryAssignLimit": 3,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config)

    for experiment in opt.get_experiments(project_name=args.experiment):
        experiment.auto_metric_logging = False
        experiment.workspace = args.workspace
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        oracle_bonus = experiment.get_parameter("oracle_bonus")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'oracle_bonus': oracle_bonus,
        })

        sac_experiment = RlhfTrackToLearnTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
