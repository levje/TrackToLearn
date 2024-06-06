import os
import argparse
import lightning.pytorch
import lightning.pytorch.trainer
import numpy as np
import tempfile
import h5py
import lightning

from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import set_number_of_points

from TrackToLearn.trainers.sac_auto_train import SACAutoTrackToLearnTraining, add_sac_auto_args
from TrackToLearn.trainers.train import add_training_args
from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.filterers.tractometer_filterer import TractometerFilterer
from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.trainers.oracle.data_module import StreamlineDataModule
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

        self.agent_checkpoint_dir = rlhf_train_dto.get('agent_checkpoint', None)
        self.num_workers = rlhf_train_dto['num_workers']
        self.comet_logger = lightning.pytorch.loggers.CometLogger(
            save_dir=self.comet_offline_dir,
            offline=self.comet_offline_dir is not None,
            project_name="TractOracleRLHF",
            experiment_name='-'.join([self.experiment, self.name]),
        )
        self.lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        self.oracle_max_ep_per_iter = 10
        self.oracle_next_max_ep = 0 # The value we add to the lightning trainer across different runs.
        self.oracle_trainer = lightning.pytorch.trainer.Trainer(
            logger=self.comet_logger,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.oracle_max_ep_per_iter,
            enable_checkpointing=True,
            default_root_dir=os.path.join(self.experiment_path, self.experiment, self.name),
            precision='16-mixed',
            callbacks=[self.lr_monitor],
            # accelerator=get_device_str(),
            # devices=1
        )

    def run(self):
        """ Prepare the environment, algorithm and trackers and run the
        training loop
        """
        super(RlhfTrackToLearnTraining, self).run()

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv
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
        
        if self.agent_checkpoint_dir is None:
            # Start by pretraining the RL agent to get reasonable results.
            super().rl_train(alg, env, valid_env)
        else:
            # The agent is already pretrained, just need to fine-tune it.
            print("Skipping pretraining procedure: loading agent from checkpoint...", end="")
            alg.agent.load(self.agent_checkpoint_dir, 'last_model_state')
            print("Done.")

        # Setup oracle training
        self.oracle = OracleSingleton(self.oracle_checkpoint,
                                      device=self.device,
                                      batch_size=self.batch_size)

        # Setup environment
        self.tracker_env = self.get_valid_env()
        self.tracker = Tracker(
            alg, self.n_actor, prob=1.0, compress=0.0)

        # Setup filterers which will be used to filter tractograms
        # for the RLHF pipeline.
        self.filterers = [
            TractometerFilterer(self.scoring_data, self.tractometer_reference, dilate_endpoints=self.tractometer_dilate)
        ]

        # RLHF loop to fine-tune the oracle to the RL agent and vice-versa.
        num_iters = 5
        for i in range(num_iters):
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

                # Create HDF5 dataset file
                dataset_file = os.path.join(tmpdir, "dataset.hdf5")
                self.create_dataset(filtered_tractograms, dataset_file)

                # Train reward model
                self.train_reward(dataset_file)

            # Train the RL agent
            super().rl_train(alg, env, valid_env)

    def train_reward(self, dataset_file: str):
        """ Train the reward model using the dataset file. """
        dm = StreamlineDataModule(dataset_file, batch_size=self.batch_size, num_workers=self.num_workers)

        # Not sure if it's the best way to iteratively train the oracle, but
        # using https://github.com/Lightning-AI/pytorch-lightning/issues/11425
        # to call fit multiple times.
        self.oracle_trainer.fit_loop.max_epochs += self.oracle_next_max_ep
        self.oracle_trainer.fit(self.oracle.model, dm)
        self.oracle_next_max_ep = self.oracle_max_ep_per_iter

        self.oracle_trainer.test(self.oracle.model, dm)


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

    def _add_streamlines_to_hdf5(self, hdf_subject, sft, nb_points, total, idx):
        """ Add the streamlines to the hdf5 file.

        TODO: THIS FUNCTION WAS COPIED FROM TRACTORACLE. MIGHT NEED TO BE REFACTORED OR ADAPTED INSTEAD OF COPIED.

        Parameters
        ----------
        hdf_subject: h5py.File
            HDF5 file to save the dataset to.
        sft: nib.streamlines.tractogram.Tractogram
            Streamlines to add to the dataset.
        nb_points: int, optional
            Number of points to resample the streamlines to
        total: int
            Total number of streamlines in the dataset
        idx: list
            List of positions to store the streamlines
        """

        # Get the scores and the streamlines
        scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)
        # Resample the streamlines
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        streamlines = np.asarray(streamlines)

        # Create the dataset if it does not exist
        if 'streamlines' not in hdf_subject:
            # Create the group
            streamlines_group = hdf_subject.create_group('streamlines')
            # Set the number of points
            streamlines = np.asarray(streamlines)
            # 'data' will contain the streamlines
            streamlines_group.create_dataset(
                'data', shape=(total, nb_points, streamlines.shape[-1]))
            # 'scores' will contain the scores
            streamlines_group.create_dataset('scores', shape=(total))

        streamlines_group = hdf_subject['streamlines']
        data_group = streamlines_group['data']
        scores_group = streamlines_group['scores']

        for i, st, sc in zip(idx, streamlines, scores):
            data_group[i] = st
            scores_group[i] = sc


    def create_dataset(self,
                       filtered_tractograms: list[StatefulTractogram],
                       out_file: str):
        """ Gathers all the filtered tractograms and creates a dataset for the reward model training. 
        Outputs into a hdf5 file."""

        # Compute the total number of streamlines
        nb_streamlines = sum([len(sft.streamlines) for sft in filtered_tractograms])
        nb_points = 128
        assert nb_streamlines > 0, "No streamlines to create the dataset."

        indices = np.arange(nb_streamlines)
        np.random.shuffle(indices)

        # Add the streamlines to the dataset
        with h5py.File(out_file, 'w') as f:
            f.attrs['version'] = 1
            f.attrs['nb_points'] = nb_points
            
            for sft in filtered_tractograms:
                sft.to_corner()
                sft.to_vox()
                sft_nb_streamlines = len(sft.streamlines)
                ps_indices = np.random.choice(sft_nb_streamlines, sft_nb_streamlines, replace=False)
                idx = indices[:sft_nb_streamlines]

                self._add_streamlines_to_hdf5(f, sft[ps_indices], nb_points, nb_streamlines, idx)

                indices = indices[sft_nb_streamlines:]
        
        return out_file


############################

def add_rlhf_training_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("RLHF Training Arguments")
    parser.add_argument('--agent_checkpoint', type=str,
                             help='Path to the folder containing .pth files.\n'
                             'This avoids retraining the agent from scratch \n'
                             'and allows to directly fine-tune it.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of workers to use for data loading.')
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

    # Create and run the experiment
    rlhf_experiment = RlhfTrackToLearnTraining(
        vars(args),
        experiment
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()