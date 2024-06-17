import os
import argparse
import lightning.pytorch
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import lightning.pytorch.trainer
import lightning.pytorch.utilities
import numpy as np
import tempfile
import h5py
import lightning
import comet_ml

from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from TrackToLearn.trainers.sac_auto_train import SACAutoTrackToLearnTraining, add_sac_auto_args
from TrackToLearn.trainers.train import add_training_args
from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.filterers.tractometer_filterer import TractometerFilterer
from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.trainers.oracle.data_module import StreamlineDataModule
from TrackToLearn.trainers.oracle.streamline_dataset_manager import StreamlineDatasetManager
from TrackToLearn.utils.torch_utils import assert_accelerator, get_device_str

assert_accelerator()

"""
In classic RLHF, the reward network is trained on predictions. Here, we will train the reward network
in the loop with the RL agent using filtered tractograms from Extractor. However, if we want to train
the reward network with multiple filtering algorithms (i.e. Tractometer, COMMIT, Extractor, etc.), we
could use the following approach to learn with "preference" data:
    
    mu(1) = smax(nb of times that streamline was classified as valid in the ensemble of filterers)
    mu(2) = smax(nb of times that streamline was classified as valid in the ensemble of filterers)

Not sure how slow/fast that would be to train in practice, but that preference distribution could be
used as a target for the reward network.

N.B: Maybe the "nb of times that streamline was classified can be a weighted sum of all the filtering
methods (e.g. more weight on Extractor than COMMIT for example)."

N.B: The pair of streamlines for comparison should originate from the same seed.

N.B: Instead of pairs, maybe a ranking system would be better, as it shows better performance in some scenarios
of the litterature.
"""

class RlhfTrackToLearnTraining(SACAutoTrackToLearnTraining):
    
    def __init__(
        self,
        rlhf_train_dto: dict,
        comet_experiment: CometExperiment,
        ):
        super().__init__(
            rlhf_train_dto,
            comet_experiment,
        )

        self.pretrain_max_ep = rlhf_train_dto.get('pretrain_max_ep', None)
        self.agent_checkpoint_dir = rlhf_train_dto.get('agent_checkpoint', None)

        # To easily compare the reference model with the trained model after training.
        self.ref_model_dir = os.path.join(self.experiment_path, "ref_model")
        self.model_saving_dirs.append(self.ref_model_dir)
        if not os.path.exists(self.ref_model_dir):
            os.makedirs(self.ref_model_dir)

        assert self.pretrain_max_ep is not None or self.agent_checkpoint_dir is not None, \
            "Either pretrain_max_ep or agent_checkpoint must be provided for RLHF training."

        self.oracle_train_steps = rlhf_train_dto['oracle_train_steps']
        self.agent_train_steps = rlhf_train_dto['agent_train_steps']
        self.num_workers = rlhf_train_dto['num_workers']
        self.rlhf_inter_npv = rlhf_train_dto['rlhf_inter_npv']
        self.disable_oracle_training = rlhf_train_dto.get('disable_oracle_training', False)

        dataset_to_augment = rlhf_train_dto.get('dataset_to_augment', None)
        self.dataset_manager = StreamlineDatasetManager(saving_path=self.model_dir,
                                                        dataset_to_augment_path=dataset_to_augment)
        
        comet_ml.config.set_global_experiment(None) # Need this to avoid erasing the RL agent's experiment
                                                    # when creating a new one.
        self.comet_logger = lightning.pytorch.loggers.CometLogger(
            save_dir=self.comet_offline_dir,
            offline=self.comet_offline_dir is not None,
            project_name="TractOracleRLHF",
            experiment_name='-'.join([self.experiment, self.name]),
        )

        self.lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        self.oracle_next_train_steps = 0 # Since we are leveraging the same trainer, we need to add
                                         # this value to the oracle trainer across different runs.
        
        self.oracle_checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=self.model_dir,
        )
        self.oracle_trainer = lightning.pytorch.trainer.Trainer(
            logger=self.comet_logger,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.oracle_train_steps,
            enable_checkpointing=True,
            default_root_dir=self.experiment_path,
            precision='16-mixed',
            callbacks=[self.lr_monitor, self.oracle_checkpoint_callback],
            accelerator=get_device_str(),
            devices=1
        )

        self.combined_train_loader = lightning.pytorch.utilities.CombinedLoader([])
        self.combined_val_loader = lightning.pytorch.utilities.CombinedLoader([])
        self.combined_test_loader = lightning.pytorch.utilities.CombinedLoader([])

    def run(self):
        """ Prepare the environment, algorithm and trackers and run the
        training loop
        """
        super(RlhfTrackToLearnTraining, self).run()

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

        if self.agent_checkpoint_dir is None:
            # Start by pretraining the RL agent to get reasonable results.
            super(RlhfTrackToLearnTraining, self).rl_train(alg,
                             env,
                             valid_env,
                             max_ep=self.pretrain_max_ep,
                             starting_ep=0,
                             save_model_dir=self.ref_model_dir)
            current_ep += self.pretrain_max_ep
        else:
            # The agent is already pretrained, just need to fine-tune it.
            print("Skipping pretraining procedure: loading agent from checkpoint...", end=" ")
            alg.agent.load(self.agent_checkpoint_dir, 'last_model_state')
            self.save_model(alg, save_model_dir=self.ref_model_dir)
            print("Done.")

        self.comet_monitor.e.add_tag("RLHF-start-ep-{}".format(current_ep))

        # Setup oracle training
        self.oracle = OracleSingleton(self.oracle_checkpoint,
                                      device=self.device,
                                      batch_size=self.batch_size)

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
                    tractograms = self.generate_and_save_tractograms(self.tracker, self.tracker_env, tractograms_path)

                    # Filter the tractogram
                    filtered_path = os.path.join(tmpdir, "filtered")
                    if not os.path.exists(filtered_path):
                        os.makedirs(filtered_path)
                    filtered_tractograms = self.filter_tractograms(tractograms, filtered_path) # Need to filter for each filterer and keep the same order.

                    self.dataset_manager.add_tractograms_to_dataset(filtered_tractograms)

                    # Train reward model
                    self.train_reward()

            # Train the RL agent
            super(RlhfTrackToLearnTraining, self).rl_train(alg,
                                                           env,
                                                           valid_env,
                                                           max_ep=self.agent_train_steps,
                                                           starting_ep=current_ep,
                                                           save_model_dir=self.model_dir)
            current_ep += self.agent_train_steps

            self.end_finetuning_epoch(i)

    def train_reward(self):
        """ Train the reward model using the dataset file. """
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path, batch_size=self.batch_size, num_workers=self.num_workers)

        dm.setup('fit')

        # TODO: To avoid using weird combination of CombinedLoader, we can also try to wrap
        # the datamodule to be able to swap it and destroy it dynamically?
        self.combined_train_loader.flattened = [dm.train_dataloader()]
        self.combined_val_loader.flattened = [dm.val_dataloader()]

        # Not sure if it's the best way to iteratively train the oracle, but
        # using https://github.com/Lightning-AI/pytorch-lightning/issues/11425
        # to call fit multiple times.
        self.oracle_trainer.fit_loop.max_epochs += self.oracle_next_train_steps # On first iteration, we add 0.
        if hasattr(self, 'experiment_key'):
            self.comet_logger._experiment_key = self.experiment_key

        self.oracle_trainer.fit(self.oracle.model, train_dataloaders=self.combined_train_loader, val_dataloaders=self.combined_val_loader)
        self.oracle_next_train_steps = self.oracle_train_steps
        self.experiment_key = self.comet_logger.experiment.get_key()

        dm.setup('test')
        self.combined_test_loader.flattened = [dm.test_dataloader()]
        self.oracle_trainer.test(self.oracle.model, dataloaders=self.combined_test_loader)

        self.combined_train_loader.flattened = [[]]
        self.combined_val_loader.flattened = [[]]
        self.combined_test_loader.flattened = [[]]

    def generate_and_save_tractograms(self, tracker: Tracker, env: BaseEnv, save_dir: str):
        tractogram, _ = tracker.track_and_validate(self.tracker_env) # TODO: Change to only track().
        filename = self.save_rasmm_tractogram(
            tractogram,
            env.subject_id,
            env.affine_vox2rasmm,
            env.reference,
            save_dir,
            extension='tck')
        
        # sft = StatefulTractogram(
        #     tractogram.streamlines,
        #     env.reference,
        #     Space.RASMM,
        #     origin=Origin.TRACKVIS,
        #     data_per_streamline=tractogram.data_per_streamline,
        #     data_per_point=tractogram.data_per_point
        # )

        # filename = "tractogram_{}_{}_{}.tck".format(self.experiment, self.name, env.subject_id)
        # save_tractogram(sft, os.path.join(save_dir, filename))

        return [filename]

    def filter_tractograms(self, tractograms: str, out_dir: str):
        """ Filter the tractogram (0 for invalid, 1 for valid) using the filterers

        TODO: Implement for more than one filterer
        """
        filterer = self.filterers[0]

        filtered_tractograms = []
        for tractogram in tractograms:
            # TODO: Implement for more than one filterer
            filtered_tractogram = filterer(tractogram, out_dir, scored_extension="trk")
            filtered_tractograms.append(filtered_tractogram)
        
        return filtered_tractograms
    
    def save_hyperparameters(self):
        """ Add SACAuto-specific hyperparameters to self.hyperparameters
        then save to file.
        """

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
             })

        super().save_hyperparameters()

    def start_finetuning_epoch(self, epoch: int):
        print("==================================================")
        print("======= Starting RLHF finetuning epoch {}/{} =======".format(epoch+1, self.max_ep))

    def end_finetuning_epoch(self, epoch: int):
        print("======= Finished RLHF finetuning epoch {}/{} =======".format(epoch+1, self.max_ep))
        print("==================================================")


############################

def add_rlhf_training_args(parser: argparse.ArgumentParser):
    rlhf_group = parser.add_argument_group("RLHF Training Arguments")

    agent_group = rlhf_group.add_mutually_exclusive_group(required=True)
    agent_group.add_argument('--pretrain_max_ep', type=int,
                        help='Number of epochs for pretraining the RL agent.\n'
                             'This is done before starting the RLHF pretraining procedure.')
    agent_group.add_argument('--agent_checkpoint', type=str,
                        help='Path to the folder containing .pth files.\n'
                             'This avoids retraining the agent from scratch \n'
                             'and allows to directly fine-tune it.')
    
    rlhf_group.add_argument('--oracle_train_steps', type=int, required=True,
                        help='Number of steps to fine-tune the oracle during RLHF training.')
    rlhf_group.add_argument('--agent_train_steps', type=int, required=True,
                        help='Number of steps to fine-tune the agent during RLHF training.')
    rlhf_group.add_argument('--num_workers', type=int, default=10,
                        help='Number of workers to use for data loading.')
    rlhf_group.add_argument("--dataset_to_augment", type=str, help="Path to the dataset to augment.\n"
                            "If this is not set, the dataset will be created from scratch entirely by the\n"
                            "current learning agent.")
    rlhf_group.add_argument("--rlhf_inter_npv", type=int, default=None,
                            help="Number of seeds to use when generating intermediate tractograms\n"
                            "for the RLHF training pipeline. If None, the general npv will be used.")
    rlhf_group.add_argument("--disable_oracle_training", action="store_true",)
    return parser

def parse_args():
    """ Train a RL tracking agent using RLHF with PPO. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_rlhf_training_args(parser)

    arguments = parser.parse_args()
    return arguments

def main():
    args = parse_args()
    
    offline = args.comet_offline_dir is not None

    # Create comet-ml experiment for agent training.
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

    # Create and run the experiment
    rlhf_experiment = RlhfTrackToLearnTraining(
        vars(args),
        experiment
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()