#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch
import os
import sys

from TrackToLearn.trainers.classic_ppo_train import (
    PPOTrackToLearnTraining,
    parse_args
)
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator
device = get_device()
assert_accelerator()


def main():
    """ Main tracking script """
    args = parse_args()
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [1e-5] # [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
            },
            "gamma": {
                "type": "discrete",
                "values": [0.5]
            }
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
        gamma = experiment.get_parameter("gamma")

        arguments = vars(args)
        path_suffix = f"lr_{str(lr).replace('.', '')}_gamma_{str(gamma).replace('.', '')}"
        arguments.update({
            'path': os.path.join(args.path, path_suffix),
            'lr': lr,
            'gamma': gamma,
        })

        sac_experiment = PPOTrackToLearnTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
