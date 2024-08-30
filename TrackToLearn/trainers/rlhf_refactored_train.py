import os
import argparse
import numpy as np
import tempfile
import h5py
import comet_ml
import json

from comet_ml import Experiment as CometExperiment

from TrackToLearn.trainers.sac_auto_train import TrackToLearnTraining, add_sac_auto_args, SACAutoTrackToLearnTraining
from TrackToLearn.trainers.ppo_train import PPOTrackToLearnTraining, add_ppo_args
from TrackToLearn.trainers.train import add_training_args
from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.filterers.tractometer_filterer import TractometerFilterer
from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.trainers.oracle.oracle_trainer import OracleTrainer
from TrackToLearn.trainers.oracle.data_module import StreamlineDataModule
from TrackToLearn.trainers.oracle.streamline_dataset_manager import StreamlineDatasetManager
from TrackToLearn.utils.torch_utils import assert_accelerator, get_device_str

assert_accelerator()

class RlhfRefactored(TrackToLearnTraining):
    
    def __init__(
        self,
        rlhf_train_dto: dict,
        trainer_cls: TrackToLearnTraining,
        agent_experiment: CometExperiment = None,
        oracle_experiment: CometExperiment = None
        ):
        super().__init__(
            rlhf_train_dto
        )

        # General RLHF parameters.
        self.pretrain_max_ep = rlhf_train_dto.get('pretrain_max_ep', None)
        self.agent_checkpoint_dir = rlhf_train_dto.get('agent_checkpoint', None)
        self.ref_model_dir = os.path.join(self.experiment_path, "ref_model")
        self.model_saving_dirs.append(self.ref_model_dir)
        if not os.path.exists(self.ref_model_dir):
            os.makedirs(self.ref_model_dir)

        self.oracle_training_dir = os.path.join(self.experiment_path, "oracle")
        if not os.path.exists(self.oracle_training_dir):
            os.makedirs(self.oracle_training_dir)

        assert self.pretrain_max_ep is not None or self.agent_checkpoint_dir is not None, \
            "Either pretrain_max_ep or agent_checkpoint must be provided for RLHF training."
        
        self.oracle_train_steps = rlhf_train_dto['oracle_train_steps']
        self.agent_train_steps = rlhf_train_dto['agent_train_steps']
        self.num_workers = rlhf_train_dto['num_workers']
        self.rlhf_inter_npv = rlhf_train_dto['rlhf_inter_npv']
        self.disable_oracle_training = rlhf_train_dto.get('disable_oracle_training', False)
        self.batch_size = rlhf_train_dto['batch_size']
        self.oracle_batch_size = rlhf_train_dto['oracle_batch_size']

        ################################################
        # Start by initializing the agent trainer.     #
        if agent_experiment is None:
            agent_experiment = CometExperiment(project_name=self.experiment,
                                        workspace=rlhf_train_dto['workspace'], parse_args=False,
                                        auto_metric_logging=False,
                                        disabled=not self.use_comet)

            agent_experiment.set_name(self.name)

        self.agent_trainer = trainer_cls(rlhf_train_dto, agent_experiment)
        _ = self.agent_trainer.setup_environment_and_info()
        self.get_alg = self.agent_trainer.get_alg
        
        ################################################
        # Continue by initializing the oracle trainer. #
        comet_ml.config.set_global_experiment(None) # Need this to avoid erasing the RL agent's experiment
                                                    # when creating a new one.
        if oracle_experiment is None:
            oracle_experiment = CometExperiment(project_name="TractOracleRLHF",
                                        workspace=rlhf_train_dto['workspace'], parse_args=False,
                                        auto_metric_logging=False,
                                        disabled=not self.use_comet)

            oracle_experiment_id = '-'.join([self.experiment, self.name])

        dataset_to_augment = rlhf_train_dto.get('dataset_to_augment', None)
        self.dataset_manager = StreamlineDatasetManager(saving_path=self.oracle_training_dir,
                                                        dataset_to_augment_path=dataset_to_augment)

        self.oracle_trainer = OracleTrainer(
            oracle_experiment,
            oracle_experiment_id,
            self.oracle_training_dir,
            self.oracle_train_steps,
            enable_checkpointing=True,
            val_interval=1,
            device=self.device
        )

        # Update the hyperparameters
        self.hyperparameters.update(
            {'algorithm': 'RLHF',
             'RL_algorithm': 'SACAuto',
             'ref_model_dir': self.ref_model_dir,
             'pretrain_max_ep': self.pretrain_max_ep,
             'agent_checkpoint_dir': self.agent_checkpoint_dir,
             'oracle_train_steps': self.oracle_train_steps,
             'agent_train_steps': self.agent_train_steps,
             'rlhf_inter_npv': self.rlhf_inter_npv,
             'oracle_training_enabled': self.disable_oracle_training,
             'original_dataset': dataset_to_augment,
             'augmented_dataset': self.dataset_manager.dataset_file_path,
             })

    def setup_logging(self):
        """ Override the setup_logging method to avoid creating a new experiment. """
        self.save_hyperparameters()

    def run(self):
        """ Prepare the environment, algorithm and trackers and run the
        training loop
        """
        super().run()

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv,
        max_ep: int = 10
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            valid_env: BaseEnv
                The validation tracking environment (forward).
            """
        assert self.oracle_checkpoint is not None, "Oracle checkpoint must be provided for RLHF training."
        assert os.path.exists(self.oracle_checkpoint), "Oracle checkpoint does not exist."
        
        current_ep = 0

        # Setup agent trainer. (needed since we don't call the run method)
        self.agent_trainer.setup_logging()

        if self.agent_checkpoint_dir is None:
            # Start by pretraining the RL agent to get reasonable results.
            if isinstance(alg, PPO):
                alg.kl_penalty_ctrler.pretrain_mode() # TODO: Refactor

            self.agent_trainer.rl_train(alg,
                             env,
                             valid_env,
                             max_ep=self.pretrain_max_ep,
                             starting_ep=0,
                             save_model_dir=self.ref_model_dir)
            current_ep += self.pretrain_max_ep
            alg.old_agent.load_state_dict(alg.agent.actor.state_dict()) # TODO: Refactor
        else:
            # The agent is already pretrained, just need to fine-tune it.
            print("Skipping pretraining procedure: loading agent from checkpoint...", end=" ")
            self._load_agent_checkpoint(alg)
            print("Done.")

        if isinstance(alg, PPO):
            alg.kl_penalty_ctrler.rlhf_mode() # TODO: Refactor

        self.agent_trainer.comet_monitor.e.add_tag("RLHF-start-ep-{}".format(current_ep))

        # Setup oracle training
        self.oracle = OracleSingleton(self.oracle_checkpoint,
                                      device=self.device,
                                      batch_size=self.oracle_batch_size)
        self.oracle_trainer.setup_model_training(self.oracle.model)
        
        # Setup environment
        self.tracker_env = self.get_valid_env(npv=self.rlhf_inter_npv)
        self.tracker = Tracker(
            alg, self.n_actor, prob=1.0, compress=0.0)
        
        # Setup filterers which will be used to filter tractograms
        # for the RLHF pipeline.
        self.filterers = [
            TractometerFilterer(self.scoring_data, self.tractometer_reference, dilate_endpoints=self.tractometer_dilate)
        ]

        # RLHF loop to fine-tune the oracle to the RL agent and vice-versa.
        for i in range(max_ep):

            self.start_finetuning_epoch(i)

            if self.disable_oracle_training:
                print("Oracle training is disabled. Only the agent will be trained and the dataset will not be augmented.\n",
                      "This is equivalent to just training the agent for an additional {} ({} x {}) epochs.".format(self.agent_train_steps*max_ep, max_ep, self.agent_train_steps))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Generate a tractogram
                    tractograms_path = os.path.join(tmpdir, "tractograms")
                    if not os.path.exists(tractograms_path):
                        os.makedirs(tractograms_path)
                    print("Generating tractograms for RLHF training...")
                    tractograms = self.generate_and_save_tractograms(self.tracker, self.tracker_env, tractograms_path)

                    # Filter the tractogram
                    filtered_path = os.path.join(tmpdir, "filtered")
                    if not os.path.exists(filtered_path):
                        os.makedirs(filtered_path)
                    print("Filtering tractograms for RLHF training...")
                    filtered_tractograms = self.filter_tractograms(tractograms, filtered_path) # Need to filter for each filterer and keep the same order.

                    print("Adding filtered tractograms to the dataset...")
                    self.dataset_manager.add_tractograms_to_dataset(filtered_tractograms)

                    # Train reward model
                    print("Training reward model...")
                    self.train_reward()

            # Train the RL agent
            self.agent_trainer.rl_train(alg,
                                        env,
                                        valid_env,
                                        max_ep=self.agent_train_steps,
                                        starting_ep=current_ep,
                                        save_model_dir=self.model_dir)
            current_ep += self.agent_train_steps

            self.end_finetuning_epoch(i)

    def train_reward(self):
        """ Train the reward model using the dataset file. """
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.oracle_batch_size,
                                  num_workers=self.num_workers)
        dm.setup('fit')
        self.oracle_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())

        dm.setup('test')
        self.oracle_trainer.test(test_dataloader=dm.test_dataloader())

    def _load_agent_checkpoint(self, alg: RLAlgorithm):
        """
        Load the agent checkpoint from which to start the RLHF fine-tuning process.
        This function handles loading the hyperparameters dictionnary and loading the
        model weights for the policy and/or the critic depending on the case.

        Several technicalities are handled here to ease the experimentation process. Different
        initialization strategies can be used depending on the algorithm used to pretrain and the
        algorithm to fine-tune the agent here. If the pretrained agent is a SACAuto agent and the
        fine-tuning algorithm is PPO, there's a difference between the architecture of the policy
        and the critic. The SAC policy maximizes entropy and predicts it's own standard deviation
        distribution while the PPO policy only predicts means (with independent std parameter).
        
        Here are the cases that are handled:

        1. Pretrain with SAC and fine-tune with SAC.
        2. Pretrain with SAC and fine-tune with PPO.
            2.1 MaxEntropyActor policy + random critic initialization.
                    In this case, the critic is randomly initialized and the PPO policy
                    is a MaxEntropyActor which is loaded from the checkpoint. Although,
                    having a MaxEntropyActor for PPO is not recommended, PPO usually has
                    an independent standard deviation parameter.
            2.2 MaxEntropyActor policy + critic initialized from reward model (with same architecture as reward model).
                    In this case, the critic's architecture is different from what is used
                    normally in our PPO implementation. The input of the critic is modified
                    to match the reward model's input, meaning that the state inputted to the
                    model is the current streamline (in other cases, the input state has the
                    information about the N previous directions and the voxels neighboring the 
                    tip of the streamline). Similarily to the previous case, having a MaxEntropyActor
                    might not be the best choice to optimize with PPO. However, here the policy
                    and the critic do not have the same input, which could be problematic.
                    NB: This approach was suggested in the InstructGPT paper:
                        (https://arxiv.org/abs/2203.02155)
        3. Pretrain with PPO and fine-tune with PPO.

        """
        def load_hyperparameters(hparams_path):
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
            return hparams

        hparams = load_hyperparameters(os.path.join(self.agent_checkpoint_dir, 'hyperparameters.json'))
        ckpt_algo = get_algorithm_cls(hparams['algorithm'])

        if isinstance(alg, ckpt_algo): # Same algorithm, same architecture.
            alg.agent.load(self.agent_checkpoint_dir, 'last_model_state')
        elif ckpt_algo == SACAuto and isinstance(alg, PPO):
            # This is needed, because PPO doesn't have the same critic as SAC.
            # This means that we start the PPO training with a randomly initialized critic.

            # Only load the policy.
            # The critic will be initialized either:
            # 1. Loaded from the checkpoint. This is done in the constructor of PPOActorCritic.
            # 2. Randomly initialized.
            alg.agent.load_policy(self.agent_checkpoint_dir, 'last_model_state')
        else:
            raise ValueError("Invalid combination of algorithms for RLHF training. Got {} and {}."
                             .format(ckpt_algo.__name__, alg.__class__.__name__))
        
        self.save_model(alg, save_model_dir=self.ref_model_dir)

    def generate_and_save_tractograms(self, tracker: Tracker, env: BaseEnv, save_dir: str):
        tractogram, _ = tracker.track_and_validate(self.tracker_env) # TODO: Change to only track().
        filename = self.save_rasmm_tractogram(
            tractogram,
            env.subject_id,
            env.affine_vox2rasmm,
            env.reference,
            save_dir,
            extension='tck')
        return [filename]

    def filter_tractograms(self, tractograms: str, out_dir: str):
        filterer = self.filterers[0]

        filtered_tractograms = []
        for tractogram in tractograms:
            # TODO: Implement for more than one filterer
            filtered_tractogram = filterer(tractogram, out_dir, scored_extension="trk")
            filtered_tractograms.append(filtered_tractogram)
        
        return filtered_tractograms
    
    def save_hyperparameters(self):
        # self.hyperparameters.update({})
        super().save_hyperparameters(filename='rlhf_hyperparameters.json')

    def start_finetuning_epoch(self, epoch: int):
        print("==================================================")
        print("======= Starting RLHF finetuning epoch {}/{} =======".format(epoch+1, self.max_ep))

    def end_finetuning_epoch(self, epoch: int):
        print("======= Finished RLHF finetuning epoch {}/{} =======".format(epoch+1, self.max_ep))
        print("==================================================")


def add_rlhf_training_args(parser: argparse.ArgumentParser):
    rlhf_group = parser.add_argument_group("RLHF Training Arguments")
    rlhf_group.add_argument('--alg', type=str, required=True,
                        help='The algorithm to use for training the agent.\n'
                             'Possible values are: SACAuto, PPO.')
    rlhf_group.add_argument('--num_workers', type=int, default=10,
                        help='Number of workers to use for data loading.')
    rlhf_group.add_argument("--rlhf_inter_npv", type=int, default=None,
                            help="Number of seeds to use when generating intermediate tractograms\n"
                            "for the RLHF training pipeline. If None, the general npv will be used.")

    # Agent training RLHF arguments 
    agent_group = rlhf_group.add_argument_group("Agent Training Arguments")
    agent_group.add_argument('--agent_train_steps', type=int, required=True,
                        help='Number of steps to fine-tune the agent during RLHF training.')

    agent_init_group = rlhf_group.add_mutually_exclusive_group(required=True)
    agent_init_group.add_argument('--pretrain_max_ep', type=int,
                        help='Number of epochs for pretraining the RL agent.\n'
                             'This is done before starting the RLHF pretraining procedure.')
    agent_init_group.add_argument('--agent_checkpoint', type=str,
                        help='Path to the folder containing .pth files.\n'
                             'This avoids retraining the agent from scratch \n'
                             'and allows to directly fine-tune it.')

    # Oracle training RLHF arguments
    oracle_group = rlhf_group.add_argument_group("Oracle Training Arguments")
    oracle_group.add_argument('--oracle_train_steps', type=int, required=True,
                        help='Number of steps to fine-tune the oracle during RLHF training.')
    oracle_group.add_argument('--oracle_batch_size', type=int, default=2816,
                        help='Batch size to use for training the oracle.')
    oracle_group.add_argument("--dataset_to_augment", type=str, help="Path to the dataset to augment.\n"
                            "If this is not set, the dataset will be created from scratch entirely by the\n"
                            "current learning agent.")
    oracle_group.add_argument("--disable_oracle_training", action="store_true",
                            help="Disable oracle training during RLHF training.\n")
    return parser

def get_trainer_cls_and_args(alg_name: str):
    trainer_map = {
        'SACAuto': SACAutoTrackToLearnTraining,
        'PPO': PPOTrackToLearnTraining,
    }

    if alg_name not in trainer_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return trainer_map[alg_name]

def get_algorithm_cls(alg_name: str):
    algorithm_map = {
        'SACAuto': SACAuto,
        'PPO': PPO,
    }

    if alg_name not in algorithm_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return algorithm_map[alg_name]

def parse_args():
    """ Train a RL tracking agent using RLHF with PPO. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_ppo_args(parser)
    add_rlhf_training_args(parser)

    arguments = parser.parse_args()
    return arguments

def main():
    args = parse_args()

    trainer_cls = get_trainer_cls_and_args(args.alg)

    # Create and run the experiment
    rlhf_experiment = RlhfRefactored(
        vars(args),
        trainer_cls
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()