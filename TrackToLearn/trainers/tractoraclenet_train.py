import argparse
import os
from comet_ml import Experiment as CometExperiment

from TrackToLearn.utils.torch_utils import assert_accelerator, get_device
from TrackToLearn.oracles.transformer_oracle import LightningLikeModule, TransformerOracle
from TrackToLearn.trainers.oracle.data_module import StreamlineDataModule
from TrackToLearn.trainers.oracle.oracle_trainer import OracleTrainer


assert_accelerator()


class TractOracleNetTraining(object):
    def __init__(self, train_dto: dict):
        # Experiment parameters
        self.experiment_path = train_dto['path']
        self.experiment_name = train_dto['experiment']
        self.id = train_dto['id']

        # Model parameters
        self.lr = train_dto['lr']
        self.oracle_train_steps = train_dto['max_ep']
        self.n_head = train_dto['n_head']
        self.n_layers = train_dto['n_layers']
        self.checkpoint = train_dto['oracle_checkpoint']

        # Data loading parameters
        self.num_workers = train_dto['num_workers']
        self.oracle_batch_size = train_dto['oracle_batch_size']

        # Data files
        self.dataset_file = train_dto['dataset_file']
        self.use_comet = train_dto['use_comet']
        self.comet_workspace = train_dto['comet_workspace']
        self.device = get_device()

    def train(self):
        root_dir = os.path.join(self.experiment_path, self.experiment_name, self.id)
        
        # Get example input to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            model = TransformerOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, self.lr)

        oracle_experiment = CometExperiment(
            project_name=self.experiment_name,
            workspace=self.comet_workspace,
            parse_args=False,
            auto_metric_logging=False,
            disabled=not self.use_comet)

        oracle_trainer = OracleTrainer(
            oracle_experiment,
            self.id,
            root_dir,
            self.oracle_train_steps,
            enable_checkpointing=True,
            val_interval=1,
            device=self.device
        )
        oracle_trainer.setup_model_training(model)
        
        # Instanciate the datamodule
        dm = StreamlineDataModule(self.dataset_file,
                                  batch_size=self.oracle_batch_size,
                                  num_workers=self.num_workers)

        dm.setup('fit')
        oracle_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())

        # Test the model
        dm.setup('test')
        oracle_trainer.test(test_dataloader=dm.test_dataloader())
    
def parse_args():
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__)
    
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('max_ep', type=int,
                        help='Number of epochs.')
    parser.add_argument('dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of encoder layers.')
    parser.add_argument('--oracle_batch_size', type=int, default=2816,
                        help='Batch size, in number of streamlines.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of workers for dataloader.')
    parser.add_argument('--oracle_checkpoint', type=str,
                        help='Path to checkpoint. If not provided, '
                             'train from scratch.')
    parser.add_argument('--comet_workspace', type=str, default='mrzarfir',
                            help='Comet workspace.')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet for logging.')

    return parser.parse_args()
    

def main():
    args = parse_args()
    training = TractOracleNetTraining(vars(args))
    training.train()

if __name__ == "__main__":
    main()
    
