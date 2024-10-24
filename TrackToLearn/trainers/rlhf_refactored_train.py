import os
import argparse
import tempfile
import comet_ml
import json
import logging

from comet_ml import Experiment as CometExperiment

from TrackToLearn.trainers.sac_auto_train import TrackToLearnTraining, add_sac_auto_args, SACAutoTrackToLearnTraining
from TrackToLearn.trainers.ppo_train import PPOTrackToLearnTraining, add_ppo_args
from TrackToLearn.trainers.tractoraclenet_train import add_oracle_train_args
from TrackToLearn.trainers.train import add_training_args
from TrackToLearn.utils.logging import setup_logging, add_logging_args
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
from TrackToLearn.utils.torch_utils import assert_accelerator
from TrackToLearn.utils.utils import prettier_metrics, prettier_dict

assert_accelerator()

LOGGER = logging.getLogger(__name__)

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
        self.agent_checkpoint_dir = rlhf_train_dto.get(
            'agent_checkpoint_dir', None)
        self.agent_checkpoint = rlhf_train_dto.get('agent_checkpoint', None)
        if self.agent_checkpoint:
            assert os.path.isfile(
                self.agent_checkpoint), "Agent checkpoint must be an checkpoint file."

        self.ref_model_dir = os.path.join(self.experiment_path, "ref_model")
        self.model_saving_dirs.append(self.ref_model_dir)
        if not os.path.exists(self.ref_model_dir):
            os.makedirs(self.ref_model_dir)

        self.oracle_training_dir = os.path.join(self.experiment_path, "oracle")
        if not os.path.exists(self.oracle_training_dir):
            os.makedirs(self.oracle_training_dir)

        assert self.pretrain_max_ep is not None or (self.agent_checkpoint_dir is not None or self.agent_checkpoint is not None), \
            "Either pretrain_max_ep or (agent_checkpoint | agent_checkpoint_dir) must be provided for RLHF training."

        self.oracle_lr = rlhf_train_dto.get('oracle_lr', None)
        self.oracle_train_steps = rlhf_train_dto['oracle_train_steps']
        self.agent_train_steps = rlhf_train_dto['agent_train_steps']
        self.num_workers = rlhf_train_dto['num_workers']
        self.rlhf_inter_npv = rlhf_train_dto['rlhf_inter_npv']
        self.disable_oracle_training = rlhf_train_dto.get(
            'disable_oracle_training', False)
        self.batch_size = rlhf_train_dto['batch_size']
        self.oracle_batch_size = rlhf_train_dto['oracle_batch_size']
        grad_accumulation_steps = rlhf_train_dto.get(
            'grad_accumulation_steps', 1)

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
        # Need this to avoid erasing the RL agent's experiment
        # when creating a new one.
        if oracle_experiment is None:
            comet_ml.config.set_global_experiment(None)
            oracle_experiment = CometExperiment(project_name="TractOracleRLHF",
                                                       workspace=rlhf_train_dto['workspace'], parse_args=False,
                                                       auto_metric_logging=False,
                                                       disabled=not self.use_comet)

            oracle_experiment_id = '-'.join([self.experiment, self.name])

        dataset_to_augment = rlhf_train_dto.get('dataset_to_augment', None)
        self.dataset_manager = StreamlineDatasetManager(saving_path=self.oracle_training_dir,
                                                        dataset_to_augment_path=dataset_to_augment)

        self.oracle_reward_trainer = OracleTrainer(
            oracle_experiment,
            oracle_experiment_id,
            self.oracle_training_dir,
            self.oracle_train_steps,
            enable_checkpointing=True,
            checkpoint_prefix='reward',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=grad_accumulation_steps,
            metrics_prefix='reward_'
        )

        self.oracle_crit_trainer = OracleTrainer(
            oracle_experiment,
            oracle_experiment_id,
            self.oracle_training_dir,
            self.oracle_train_steps,
            enable_checkpointing=True,
            checkpoint_prefix='crit',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=grad_accumulation_steps,
            metrics_prefix='crit_'
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
        assert self.oracle_crit_checkpoint is not None, "Oracle criterion" \
            "checkpoint must be provided for RLHF training."
        assert self.oracle_reward_checkpoint is not None, "Oracle reward" \
            "checkpoint must be provided for RLHF training."
        assert os.path.exists(
            self.oracle_crit_checkpoint), "Oracle criterion checkpoint" \
            "does not exist."
        assert os.path.exists(
            self.oracle_reward_checkpoint), "Oracle reward checkpoint"\
            "does not exist."

        current_ep = 0

        # Setup agent trainer. (needed since we don't call the run method)
        self.agent_trainer.setup_logging()

        if self.agent_checkpoint_dir is None and self.agent_checkpoint is None:
            # Start by pretraining the RL agent to get reasonable results.
            if isinstance(alg, PPO):
                alg.kl_penalty_ctrler.pretrain_mode()  # TODO: Refactor

            self.agent_trainer.rl_train(alg,
                                        env,
                                        valid_env,
                                        max_ep=self.pretrain_max_ep,
                                        starting_ep=0,
                                        save_model_dir=self.ref_model_dir)
            current_ep += self.pretrain_max_ep

            if isinstance(alg, PPO):
                alg.old_agent.load_state_dict(
                    alg.agent.actor.state_dict())  # TODO: Refactor
        else:
            # The agent is already pretrained, just need to fine-tune it.
            LOGGER.info(
                "Skipping pretraining procedure: loading agent from checkpoint...")
            self._load_agent_checkpoint(alg)
            LOGGER.info("Done.")

        if isinstance(alg, PPO):
            alg.kl_penalty_ctrler.rlhf_mode()  # TODO: Refactor

        self.agent_trainer.comet_monitor.e.add_tag(
            "RLHF-start-ep-{}".format(current_ep))

        # Setup oracle training

        # Load reward oracle
        self.oracle_reward = OracleSingleton(self.oracle_reward_checkpoint,
                                      device=self.device,
                                      batch_size=self.oracle_batch_size,
                                      lr=self.oracle_lr)
        self.oracle_reward_trainer.setup_model_training(self.oracle_reward.model)

        # Load stopping criterion oracle
        self.oracle_crit = OracleSingleton(self.oracle_crit_checkpoint,
                                           device=self.device,
                                           batch_size=self.oracle_batch_size,
                                           lr=self.oracle_lr)
        self.oracle_crit_trainer.setup_model_training(self.oracle_crit.model)

        # Setup environment
        self.tracker_env = self.get_valid_env(npv=self.rlhf_inter_npv)
        self.tracker = Tracker(
            alg, self.n_actor, prob=1.0, compress=0.0)

        # Setup filterers which will be used to filter tractograms
        # for the RLHF pipeline.
        self.filterers = [
            TractometerFilterer(self.scoring_data, self.tractometer_reference,
                                dilate_endpoints=self.tractometer_dilate)
        ]

        # RLHF loop to fine-tune the oracle to the RL agent and vice-versa.
        for i in range(max_ep):

            self.start_finetuning_epoch(i)

            if self.disable_oracle_training:
                LOGGER.info("Oracle training is disabled. Only the agent will be trained and the dataset will not be augmented.\n",
                                 "This is equivalent to just training the agent for an additional {} ({} x {}) epochs.".format(self.agent_train_steps*max_ep, max_ep, self.agent_train_steps))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Generate a tractogram
                    tractograms_path = os.path.join(tmpdir, "tractograms")
                    if not os.path.exists(tractograms_path):
                        os.makedirs(tractograms_path)
                    LOGGER.info(
                        "Generating tractograms for RLHF training...")
                    tractograms = self.generate_and_save_tractograms(
                        self.tracker, self.tracker_env, tractograms_path)

                    # Filter the tractogram
                    filtered_path = os.path.join(tmpdir, "filtered")
                    if not os.path.exists(filtered_path):
                        os.makedirs(filtered_path)
                    LOGGER.info(
                        "Filtering tractograms for RLHF training...")
                    # Need to filter for each filterer and keep the same order.
                    filtered_tractograms = self.filter_tractograms(
                        tractograms, filtered_path)

                    LOGGER.info(
                        "Adding filtered tractograms to the dataset...")
                    self.dataset_manager.add_tractograms_to_dataset(
                        filtered_tractograms)
                    data_stats = self.dataset_manager.fetch_dataset_stats()
                    LOGGER.info(
                        prettier_dict(data_stats, title="Dataset stats (iter {})".format(i)))

                    # Train reward model
                    LOGGER.info("Training reward model...")
                    self.train_reward()
                    self.train_stopping_criterion()

            # Train the RL agent
            self.agent_trainer.rl_train(alg,
                                        env,
                                        valid_env,
                                        max_ep=self.agent_train_steps,
                                        starting_ep=current_ep,
                                        save_model_dir=self.model_dir,
                                        test_before_training=False
                                        )
            current_ep += self.agent_train_steps

            self.end_finetuning_epoch(i)

    def train_reward(self):
        """
        Train the reward model using the dataset file.
        This reward model should have been trained on full streamlines, which
        means that dense=False and partial=False.
        """
        print(">>> Training reward model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.oracle_batch_size,
                                  num_workers=self.num_workers)
        

        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=False, partial=False)
        self.oracle_reward_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())

        metrics_after = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished training reward model step <<<")

    def train_stopping_criterion(self):
        """
        Train the stopping criterion oracle model using the dataset file.
        This stopping criterion model should have been trained on cut
        streamlines, which means that dense=True and partial=False.
        """
        print(">>> Training stopping criterion model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.oracle_batch_size,
                                  num_workers=self.num_workers)
        
        # Test the performance of the actual model BEFORE fine-tuning.
        dm.setup('test', dense=True, partial=False)
        metrics_before = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=True, partial=False)
        self.oracle_crit_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())
        
        # Test the performance of the actual model AFTER fine-tuning.
        metrics_after = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished stopping criterion model training <<<")

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

        if self.agent_checkpoint:
            # In the case we only give the bundled checkpoint file.
            # We need to extract the hyperparameters.json file which
            # contains the hyperparameters of the training and should
            # be located beside the checkpoint file.
            checkpoint_dir = os.path.dirname(self.agent_checkpoint)
        else:
            # We provide the checkpoint directory (with two files for
            # the weights of the policy and the critic).
            checkpoint_dir = self.agent_checkpoint_dir

        LOGGER.debug("Agent checkpoint: {}".format(self.agent_checkpoint))
        LOGGER.debug("checkpoint_dir: {}".format(checkpoint_dir))

        hparams = load_hyperparameters(os.path.join(
            checkpoint_dir, 'hyperparameters.json'))
        ckpt_algo = get_algorithm_cls(hparams['algorithm'])

        if isinstance(alg, ckpt_algo):  # Same algorithm, same architecture.
            if self.agent_checkpoint_dir:
                # Use the legacy method that loads the two files of weights
                # for the policy and the critic.
                alg.agent.load(self.agent_checkpoint_dir, 'last_model_state')
            elif self.agent_checkpoint:
                # Load the bundled checkpoint file.
                alg.load_checkpoint(self.agent_checkpoint)
            else:
                raise ValueError(
                    "Must specify either agent_checkpoint_dir or agent_checkpoint.")

        elif ckpt_algo == SACAuto and isinstance(alg, PPO):
            # This is needed, because PPO doesn't have the same critic as SAC.
            # This means that we start the PPO training with a randomly initialized critic.

            # Only load the policy.
            # The critic will be initialized either:
            # 1. Loaded from the checkpoint. This is done in the constructor of PPOActorCritic.
            # 2. Randomly initialized.
            alg.agent.load_policy(
                self.agent_checkpoint_dir, 'last_model_state')
        else:
            raise ValueError("Invalid combination of algorithms for RLHF training. Got {} and {}."
                             .format(ckpt_algo.__name__, alg.__class__.__name__))

        self.save_model(alg, save_model_dir=self.ref_model_dir)

    def generate_and_save_tractograms(self, tracker: Tracker, env: BaseEnv, save_dir: str):
        # TODO: Change to only track().
        tractogram, _ = tracker.track_and_validate(self.tracker_env)
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
            valid_tractogram, invalid_tractogram = filterer(
                tractogram, out_dir, scored_extension="trk")
            filtered_tractograms.append((valid_tractogram, invalid_tractogram))

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

    # The following arguments are usually used for PPO, but we are also testing it for other algorithms.
    parser.add_argument('--adaptive_kl', action='store_true',
                        help='This flag enables the adaptive kl penalty.\n'
                        'Otherwise, the penalty coefficient is fixed.')
    parser.add_argument('--kl_penalty_coeff', default=0.02, type=float,
                        help='Initial KL penalty coefficient.')
    parser.add_argument('--kl_target', default=0.005, type=float,
                        help='KL target value.')
    parser.add_argument('--kl_horizon', default=1000, type=int,
                        help='KL penalty horizon.')

    # Agent training RLHF arguments
    agent_group = rlhf_group.add_argument_group("Agent Training Arguments")
    agent_group.add_argument('--agent_train_steps', type=int, required=True,
                             help='Number of steps to fine-tune the agent during RLHF training.')

    agent_init_group = rlhf_group.add_mutually_exclusive_group(required=True)
    agent_init_group.add_argument('--pretrain_max_ep', type=int,
                                  help='Number of epochs for pretraining the RL agent.\n'
                                  'This is done before starting the RLHF pretraining procedure.')
    agent_checkpoint_group = agent_init_group.add_mutually_exclusive_group()
    agent_checkpoint_group.add_argument('--agent_checkpoint_dir', type=str,
                                        help='Path to the folder containing .pth files.\n'
                                        'This avoids retraining the agent from scratch \n'
                                        'and allows to directly fine-tune it.')
    agent_checkpoint_group.add_argument('--agent_checkpoint', type=str,
                                        help='Path to the agent checkpoint FILE to load.')

    # Oracle training RLHF arguments
    oracle_group = rlhf_group.add_argument_group("Oracle Training Arguments")
    oracle_group.add_argument('--oracle_lr', type=float,
                              help='Learning rate to use for training the oracle.\n'
                              'If not set, the lr stored in the checkpoint will be used.')
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
    add_oracle_train_args(parser)
    add_logging_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_args()
    setup_logging(args)

    trainer_cls = get_trainer_cls_and_args(args.alg)

    # Create and run the experiment
    rlhf_experiment = RlhfRefactored(
        vars(args),
        trainer_cls
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()
