#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from TrackToLearn.algorithms.sac_auto import SACAuto, SACAutoHParams
from TrackToLearn.trainers.train import (TrackToLearnTraining,
                                         add_training_args)
from TrackToLearn.utils.torch_utils import get_device
device = get_device()


class SACAutoTrackToLearnTraining(TrackToLearnTraining):
    """
    Train a RL tracking agent using SAC with automatic entropy adjustment.
    """

    def __init__(
        self,
        sac_auto_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        sac_auto_train_dto: dict
        SACAuto training parameters
        comet_experiment: CometExperiment
        Allows for logging and experiment management.
        """

        super().__init__(
            sac_auto_train_dto,
            comet_experiment,
        )

        # SACAuto-specific parameters
        self.alpha = sac_auto_train_dto['alpha']
        self.batch_size = sac_auto_train_dto['batch_size']
        self.replay_size = sac_auto_train_dto['replay_size']
        self.adaptive_kl = sac_auto_train_dto.get('adaptive_kl', None)
        self.kl_penalty_coeff = sac_auto_train_dto.get('kl_penalty_coeff', None)
        self.kl_target = sac_auto_train_dto.get('kl_target', None)
        self.kl_horizon = sac_auto_train_dto.get('kl_horizon', None)

        kl_kwargs = {}
        if self.adaptive_kl:
            kl_kwargs = {
                'adaptive_kl': self.adaptive_kl,
                'kl_penalty_coeff': self.kl_penalty_coeff,
                'kl_target': self.kl_target,
                'kl_horizon': self.kl_horizon
            }

        self.sac_auto_hparams = SACAutoHParams(
            self.lr,
            self.gamma,
            self.n_actor,
            self.alpha,
            self.batch_size,
            self.replay_size,
            **kl_kwargs
        )

    def save_hyperparameters(self):
        """ Add SACAuto-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'SACAuto',
             'alpha': self.alpha,
             'batch_size': self.batch_size,
             'replay_size': self.replay_size})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = SACAuto(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.sac_auto_hparams,
            self.rng,
            device)
        return alg


def add_sac_auto_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Initial temperature parameter')
    parser.add_argument('--batch_size', default=2**12, type=int,
                        help='How many tuples to sample from the replay '
                        'buffer.')
    parser.add_argument('--replay_size', default=1e6, type=int,
                        help='How many tuples to store in the replay buffer.')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_training_args(parser)
    add_sac_auto_args(parser)

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
    sac_auto_experiment = SACAutoTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    sac_auto_experiment.run()


if __name__ == '__main__':
    main()
