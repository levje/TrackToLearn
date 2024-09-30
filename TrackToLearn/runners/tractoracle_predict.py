import argparse
import os
from comet_ml import Experiment as CometExperiment
import torch

from TrackToLearn.utils.torch_utils import assert_accelerator, get_device
from TrackToLearn.oracles.transformer_oracle import TransformerOracle
from TrackToLearn.trainers.oracle.data_module import StreamlineDataModule
from TrackToLearn.trainers.oracle.oracle_trainer import OracleTrainer
from TrackToLearn.utils.utils import prettier_metrics


assert_accelerator()


class TractOracleNetPredict(object):
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
        self.grad_accumulation_steps = train_dto['grad_accumulation_steps']

        # Data files
        self.dataset_file = train_dto['dataset_file']
        self.use_comet = train_dto['use_comet']
        self.comet_workspace = train_dto['comet_workspace']
        self.device = get_device()

    def test(self):
        root_dir = os.path.join(self.experiment_path,
                                self.experiment_name, self.id)

        # Create the root directory if it doesn't exist
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Get example input to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            checkpoint = torch.load(self.checkpoint)
            model = TransformerOracle.load_from_checkpoint(checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, self.lr)

        print("Creating Comet experiment {} at workspace {}...".format(
            self.experiment_name, self.comet_workspace), end=' ')
        oracle_experiment = CometExperiment(
            project_name=self.experiment_name,
            workspace=self.comet_workspace,
            parse_args=False,
            auto_metric_logging=False,
            disabled=True)

        print("Done.")

        oracle_trainer = OracleTrainer(
            oracle_experiment,
            self.id,
            root_dir,
            self.oracle_train_steps,
            enable_checkpointing=True,
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=self.grad_accumulation_steps
        )
        oracle_trainer.setup_model_training(model)

        # Instanciate the datamodule
        dm = StreamlineDataModule(self.dataset_file,
                                  batch_size=self.oracle_batch_size,
                                  num_workers=self.num_workers)

        # Test the model
        dm.setup('test')
        test_metrics = oracle_trainer.test(
            test_dataloader=dm.test_dataloader())
        print("Performance on the test set:\n",
              prettier_metrics(test_metrics))


def add_oracle_train_args(parser):
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps. This is useful '
                             'when the batch size is too large to fit in memory, but'
                             'you still want to simulate a large batch size. The'
                             'grads are accumulated over the specified number of steps'
                             'before updating the weights.')


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

    add_oracle_train_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    predictor = TractOracleNetPredict(vars(args))
    predictor.test()


if __name__ == "__main__":
    main()
