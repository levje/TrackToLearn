#!/usr/bin/env python3
import argparse
import json
import nibabel as nib
import numpy as np
import os
import random
import torch

from argparse import RawTextHelpFormatter
from os.path import join

from dipy.io.utils import get_reference_info, create_tractogram_header
from nibabel.streamlines import detect_format
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             parse_sh_basis_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import verify_streamline_length_options

from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.datasets.utils import MRIDataVolume

from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.utils.torch_utils import get_device
from TrackToLearn.environments.rollout_env import RolloutEnvironment
from TrackToLearn.oracles.oracle import OracleSingleton

# Define the example model paths from the install folder.
# Hackish ? I'm not aware of a better solution but I'm
# open to suggestions.
_ROOT = os.sep.join(os.path.normpath(
    os.path.dirname(__file__)).split(os.sep)[:-2])
DEFAULT_MODEL = os.path.join(
    _ROOT, 'models')


class TrackToLearnTrack(Experiment):
    """ TrackToLearn testing script. Should work on any model trained with a
    TrackToLearn experiment
    """

    def __init__(
        self,
        track_dto,
    ):
        """
        """

        self.in_odf = track_dto['in_odf']
        self.wm_file = track_dto['in_mask']

        self.in_seed = track_dto['in_seed']
        self.in_mask = track_dto['in_mask']
        self.input_wm = track_dto['input_wm']

        self.dataset_file = None
        self.subject_id = None

        self.reference_file = track_dto['in_mask']
        self.out_tractogram = track_dto['out_tractogram']

        self.noise = track_dto['noise']

        self.binary_stopping_threshold = \
            track_dto['binary_stopping_threshold']

        self.n_actor = track_dto['n_actor']
        self.npv = track_dto['npv']
        self.min_length = track_dto['min_length']
        self.max_length = track_dto['max_length']

        self.compress = track_dto['compress'] or 0.0
        (self.sh_basis, self.is_sh_basis_legacy) = parse_sh_basis_arg(argparse.Namespace(**track_dto))
        self.save_seeds = track_dto['save_seeds']

        # Tractometer parameters
        self.tractometer_validator = False
        self.scoring_data = None

        self.compute_reward = False
        self.use_classic_reward = False
        self.render = False
        self.reward_with_gt = False

        self.device = get_device()

        self.fa_map = None
        if 'fa_map_file' in track_dto:
            fa_image = nib.load(
                track_dto['fa_map_file'])
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        self.agent_checkpoint = track_dto['agent_checkpoint']
        self.agent_checkpoint_dir = track_dto['agent_checkpoint_dir']

        def load_hyperparameters(hparams_path):
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
            return hparams
        
        checkpoint_dir = self.agent_checkpoint_dir or os.path.dirname(self.agent_checkpoint)
        
        self.hparams = load_hyperparameters(os.path.join(
            checkpoint_dir, 'hyperparameters.json'))

        self.algorithm = self.hparams['algorithm']
        self.step_size = float(self.hparams['step_size'])
        self.voxel_size = self.hparams.get('voxel_size', 2.0)
        self.theta = self.hparams['max_angle']
        self.hidden_dims = self.hparams['hidden_dims']
        self.n_dirs = self.hparams['n_dirs']
        self.target_sh_order = self.hparams['target_sh_order']

        self.alignment_weighting = 0.0
        # Oracle parameters
        self.oracle_reward_checkpoint = None
        self.oracle_crit_checkpoint = None
        self.oracle_bonus = 0
        self.oracle_validator = False
        self.oracle_stopping_criterion = False

        # Monte Carlo Oracle
        self.mc_oracle_checkpoint = track_dto['mc_oracle_checkpoint']

        self.random_seed = track_dto['rng_seed']
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.rng = np.random.RandomState(seed=self.random_seed)

        self.comet_experiment = None

    def run(self):
        """
        Main method where the magic happens
        """
        # Presume iso vox
        ref_img = nib.load(self.reference_file)
        tracking_voxel_size = ref_img.header.get_zooms()[0]

        # # Set the voxel size so the agent traverses the same "quantity" of
        # # voxels per step as during training.
        step_size_mm = self.step_size
        if abs(float(tracking_voxel_size) - float(self.voxel_size)) >= 0.1:
            step_size_mm = (
                float(tracking_voxel_size) / float(self.voxel_size)) * \
                self.step_size

            print("Agent was trained on a voxel size of {}mm and a "
                  "step size of {}mm.".format(self.voxel_size, self.step_size))

            print("Subject has a voxel size of {}mm, setting step size to "
                  "{}mm.".format(tracking_voxel_size, step_size_mm))

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        env = self.get_tracking_env()
        env.step_size_mm = step_size_mm

        # Get example state to define NN input size
        example_state, _ = env.reset(0, 1)
        self.input_size = example_state.shape[1]
        self.action_size = env.get_action_size()

        # Load agent
        algs = {'SACAuto': SACAuto, 'PPO': PPO}

        rl_alg = algs[self.algorithm]
        print('Tracking with {} agent.'.format(self.algorithm))
        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            rng=self.rng,
            device=self.device)

        # Load pretrained policies
        if self.agent_checkpoint_dir:
            # Use the legacy method that loads the two files of weights
            # for the policy and the critic.
            alg.agent.load(self.agent_checkpoint_dir, 'last_model_state')
        elif self.agent_checkpoint:
            # Load the bundled checkpoint file.
            alg.load_checkpoint(self.agent_checkpoint)

        if self.mc_oracle_checkpoint:
            oracle = OracleSingleton(self.mc_oracle_checkpoint, device=self.device)
            rollout_env = RolloutEnvironment(
                ref_img, oracle,
                min_streamline_steps=env.min_nb_steps + 1,
                max_streamline_steps=env.max_nb_steps + 1)
            rollout_env.setup_rollout_agent(alg.agent)

        # Initialize Tracker, which will handle streamline generation

        tracker = Tracker(
            alg, self.n_actor, compress=self.compress,
            min_length=self.min_length, max_length=self.max_length,
            save_seeds=self.save_seeds)

        # Run tracking
        env.load_subject()

        if self.mc_oracle_checkpoint:
            env.setup_rollout_env(rollout_env)
        
        filetype = detect_format(self.out_tractogram)
        tractogram = tracker.track(env, filetype)

        reference = get_reference_info(self.reference_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, self.out_tractogram, header=header)


def add_mandatory_options_tracking(p):
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.\nCan be of any order and basis (including "full'
                        '" bases for\nasymmetric ODFs). See also --sh_basis.')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz). Must be represent the WM/GM '
                        'interface.')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('--input_wm', action='store_true',
                   help='If set, append the WM mask to the input signal. The '
                        'agent must have been trained accordingly.')


def add_out_options(p):
    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--compress', type=float, metavar='thresh',
                       help='If set, will compress streamlines. The parameter '
                            'value is the \ndistance threshold. A rule of '
                            'thumb is to set it to 0.1mm for \ndeterministic '
                            'streamlines and 0.2mm for probabilitic '
                            'streamlines [%(default)s].')
    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scilpy_compute_seed_density_map.')
    return out_g


def add_track_args(parser):

    add_mandatory_options_tracking(parser)

    basis_group = parser.add_argument_group('Basis options')
    add_sh_basis_args(basis_group)
    add_out_options(parser)

    agent_group = parser.add_argument_group('Tracking agent options')
    agent_checkpoint_group = agent_group.add_mutually_exclusive_group(required=True)
    agent_checkpoint_group.add_argument('--agent_checkpoint_dir', type=str,
                                        help='Path to the folder containing .pth files.\n'
                                        'This avoids retraining the agent from scratch \n'
                                        'and allows to directly fine-tune it.')
    agent_checkpoint_group.add_argument('--agent_checkpoint', type=str,
                                        help='Path to the agent checkpoint FILE to load.')

    agent_group.add_argument('--n_actor', type=int, default=10000, metavar='N',
                             help='Number of streamlines to track simultaneous'
                             'ly.\nLimited by the size of your GPU and RAM. A '
                             'higher value\nwill speed up tracking up to a '
                             'point [%(default)s].')

    seed_group = parser.add_argument_group('Seeding options')
    seed_group.add_argument('--npv', type=int, default=1,
                            help='Number of seeds per voxel [%(default)s].')
    track_g = parser.add_argument_group('Tracking options')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--noise', default=0.0, type=float, metavar='sigma',
                         help='Add noise ~ N (0, `noise`) to the agent\'s\n'
                         'output to make tracking more probabilistic.\n'
                         'Should be between 0.0 and 0.1.'
                         '[%(default)s]')
    track_g.add_argument('--fa_map', type=str, default=None,
                         help='Scale the added noise (see `--noise`) according'
                         '\nto the provided FA map (.nii.gz). Optional.')
    track_g.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
             'Tracking will stop below this threshold.')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Random number generator seed [%(default)s].')

def add_monte_carlo_args(parser):
    parser.add_argument('--mc_oracle_checkpoint', type=str,
                        help='Path to the oracle checkpoint FILE to load.\n'
                        'This oracle will be used to evaluate the streamlines.'
                        '\n It should be able to predict streamlines at \n'
                        'any length.')

def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_track_args(parser)
    add_monte_carlo_args(parser)

    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_odf, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)

    return args


def main():
    """ Main tracking script """
    args = parse_args()

    experiment = TrackToLearnTrack(
        vars(args)
    )

    experiment.run()


if __name__ == '__main__':
    main()
