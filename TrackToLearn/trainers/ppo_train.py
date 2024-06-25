#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.trainers.train import TrackToLearnTraining, add_training_args
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator

assert_accelerator()

class PPOTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        ppo_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        ppo_train_dto: dict
            PPO training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            ppo_train_dto,
            comet_experiment,
        )

        # PPO-specific parameters
        self.action_std = ppo_train_dto['action_std']
        self.lmbda = ppo_train_dto['lmbda']
        self.eps_clip = ppo_train_dto['eps_clip']
        self.K_epochs = ppo_train_dto['K_epochs']
        self.entropy_loss_coeff = ppo_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add PPO-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'PPO',
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'eps_clip': self.eps_clip,
             'K_epochs': self.K_epochs,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        # The RL training algorithm
        alg = PPO(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.K_epochs,
            self.eps_clip,
            self.entropy_loss_coeff,
            max_nb_steps,
            self.n_actor,
            self.rng,
            get_device())
        return alg


def add_ppo_args(parser):
    parser.add_argument('--entropy_loss_coeff', default=0.0001, type=float,
                        help='Entropy bonus coefficient')
    parser.add_argument('--action_std', default=0.0, type=float,
                        help='Standard deviation used of the action')
    parser.add_argument('--lmbda', default=0.95, type=float,
                        help='Lambda param for advantage discounting')
    parser.add_argument('--K_epochs', default=10, type=int,
                        help='Train the model for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float,
                        help='Clipping parameter for PPO')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_training_args(parser)
    add_ppo_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    offline = args.comet_offline_dir is not None

    # Create comet-ml experiment
    if offline:
        experiment = CometOfflineExperiment(project_name=args.experiment,
                                    workspace=args.workspace, parse_args=False,
                                    auto_metric_logging=False,
                                    disabled=not args.use_comet,
                                    offline_directory=args.comet_offline_dir)
    else:
        experiment = CometExperiment(project_name=args.experiment,
                                    workspace=args.workspace, parse_args=False,
                                    auto_metric_logging=False,
                                    disabled=not args.use_comet)
    experiment.set_name(args.id)

    # Create and run experiment
    ppo_experiment = PPOTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    ppo_experiment.run()


if __name__ == '__main__':
    main()