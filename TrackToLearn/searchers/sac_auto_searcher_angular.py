#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.trainers.sac_auto_train import (
    parse_args,
    SACAutoTrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


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
                "values": [5e-4]},
            "gamma": {
                "type": "discrete",
                "values": [0.5, 0.75, 0.85, 0.9, 0.95, 0.99]},
            "alpha": {
                "type": "discrete",
                "values": [0.2]},
            "epsilon": {
                "type": "discrete",
                "values": [90, 45, 30, 15]},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
            "seed": args.rng_seed,
            "retryAssignLimit": 3,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, project_name=args.experiment)

    for experiment in opt.get_experiments():
        experiment.auto_metric_logging = False
        experiment.workspace = args.workspace
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        gamma = experiment.get_parameter("gamma")
        alpha = experiment.get_parameter("alpha")
        epsilon = experiment.get_parameter("epsilon")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma,
            'alpha': alpha,
            'epsilon': epsilon,
        })

        sac_experiment = SACAutoTrackToLearnTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()