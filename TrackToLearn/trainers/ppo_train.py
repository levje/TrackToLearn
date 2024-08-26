#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import warnings
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from TrackToLearn.algorithms.ppo import PPO, PPOHParams
from TrackToLearn.algorithms.shared.onpolicy import OracleBasedCritic, Critic
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
        self.val_clip_coef = ppo_train_dto['val_clip_coef']
        
        self.adaptive_kl = ppo_train_dto['adaptive_kl']
        self.kl_penalty_coeff = ppo_train_dto['kl_penalty_coeff']
        self.kl_target = ppo_train_dto['kl_target']
        self.kl_horizon = ppo_train_dto['kl_horizon']

        self.critic_checkpoint = None
        if ppo_train_dto['init_critic_to_oracle']:
            if self.oracle_checkpoint:
                self.critic_checkpoint = torch.load(ppo_train_dto['oracle_checkpoint'],\
                                                    map_location=get_device())
            else:
                warnings.warn("No oracle checkpoint provided, but init_critic_to_oracle is set to True.\n"
                              "Critic will be initialized randomly.")
        self.critic_architecture = OracleBasedCritic if self.critic_checkpoint is not None else Critic

        self.ppo_hparams = PPOHParams(
            self.oracle_bonus,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.K_epochs,
            self.eps_clip,
            self.val_clip_coef,
            self.entropy_loss_coeff,
            self.adaptive_kl,
            self.kl_penalty_coeff,
            self.kl_target,
            self.kl_horizon
        )

    def save_hyperparameters(self):
        """ Add PPO-specific hyperparameters to self.hyperparameters
        then save to file.
        """
        self.hyperparameters.update(self.ppo_hparams.__dict__)
        self.hyperparameters.update({
            "critic_architecture": self.critic_architecture.__name__
        })

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):

        # The RL training algorithm
        alg = PPO(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.ppo_hparams,
            self.action_std,
            max_nb_steps,
            self.n_actor,
            self.critic_checkpoint,
            self.rng,
            get_device())
        return alg


def add_ppo_args(parser):
    parser.add_argument('--entropy_loss_coeff', default=0.0001, type=float,
                        help='Entropy bonus coefficient')
    parser.add_argument('--action_std', default=0.0, type=float,
                        help='Standard deviation used of the action')
    parser.add_argument('--lmbda', default=0.95, type=float,
                        help='Lambda param for advantage discounting. 0.0 means no discounting, thus adv = R - V(s).')
    parser.add_argument('--K_epochs', default=50, type=int,
                        help='Train the model for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float,
                        help='Clipping parameter for PPO')
    parser.add_argument('--val_clip_coef', default=0.2, type=float,
                        help='Clipping parameter for the value function.')
    
    parser.add_argument('--adaptive_kl', action='store_true',
                        help='This flag enables the adaptive kl penalty.\n'
                        'Otherwise, the penalty coefficient is fixed.')
    parser.add_argument('--kl_penalty_coeff', default=0.02, type=float,
                        help='Initial KL penalty coefficient.')
    parser.add_argument('--kl_target', default=0.005, type=float,
                        help='KL target value.')
    parser.add_argument('--kl_horizon', default=1000, type=int,
                        help='KL penalty horizon.')
    parser.add_argument('--init_critic_to_oracle', action='store_true',
                        help='Initialize the critic to the oracle model.')


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