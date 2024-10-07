import math
import torch

from torch import nn, Tensor
from torchmetrics.regression import (MeanSquaredError, MeanAbsoluteError)
from torchmetrics.classification import (BinaryRecall, BinaryPrecision, BinaryAccuracy, BinaryROC,
                                         BinarySpecificity, BinaryF1Score)
from dipy.tracking.utils import length
import numpy as np
from collections import defaultdict


class LightningLikeModule(nn.Module):
    def __init__(self):
        super(LightningLikeModule, self).__init__()

    def configure_optimizers(self):
        raise NotImplementedError()

    @torch.autocast(device_type='cuda')
    def forward():
        raise NotImplementedError()

    def load_from_checkpoint():
        raise NotImplementedError()

    def training_step():
        raise NotImplementedError()

    def validation_step():
        raise NotImplementedError()

    def test_step():
        raise NotImplementedError()


class PositionalEncoding(nn.Module):
    """ From
    https://pytorch.org/tutorials/beginner/transformer_tutorial.htm://pytorch.org/tutorials/beginner/transformer_tutorial.html  # noqa E504
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x


class TransformerOracle(LightningLikeModule):

    def __init__(
            self,
            input_size,
            output_size,
            n_head,
            n_layers,
            lr,
            loss=nn.MSELoss,
            mixed_precision=True
    ):
        super(TransformerOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.n_head = n_head
        self.n_layers = n_layers
        self.enable_amp = mixed_precision

        self.embedding_size = 32

        # Class token, initialized randomly
        self.cls_token = nn.Parameter(torch.randn((3)))

        # Embedding layer
        self.embedding = nn.Sequential(
            *(nn.Linear(3, self.embedding_size),
              nn.ReLU()))

        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(
            self.embedding_size, max_len=(input_size//3) + 1)

        # Transformer encoder layer
        layer = nn.TransformerEncoderLayer(
            self.embedding_size, n_head, batch_first=True)

        # Transformer encoder
        self.bert = nn.TransformerEncoder(layer, self.n_layers)
        # Linear layer
        self.head = nn.Linear(self.embedding_size, output_size)
        # Sigmoid layer
        self.sig = nn.Sigmoid()

        # Loss function
        self.loss = loss()

        # Metrics
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.spec = BinarySpecificity()
        self.precision = BinaryPrecision()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.roc = BinaryROC()
        self.f1 = BinaryF1Score()

        # Save the hyperparameters to the checkpoint
        # self.save_hyperparameters()

    def configure_optimizers(self, trainer, checkpoint=None):
        self.trainer = trainer

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.trainer.max_epochs
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint.keys():
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

        elif hasattr(self, 'checkpoint_state_dicts') and self.checkpoint_state_dicts is not None:
            optimizer.load_state_dict(
                self.checkpoint_state_dicts["optimizer_state_dict"])
            scheduler.load_state_dict(
                self.checkpoint_state_dicts["scheduler_state_dict"])

            if "scaler_state_dict" in self.checkpoint_state_dicts.keys():
                scaler.load_state_dict(
                    self.checkpoint_state_dicts["scaler_state_dict"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
            "scaler": scaler
        }

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.squeeze(0)
        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.embedding(x) * math.sqrt(self.embedding_size)

        encoding = self.pos_encoding(x)

        hidden = self.bert(encoding)

        y = self.head(hidden[:, 0])

        if self.loss is not nn.BCEWithLogitsLoss:
            y = self.sig(y)
        else:
            y = hidden[:, 0]

        return y.squeeze(-1)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict, lr: float = None):
        # Checkpoint is a dict with the following structure:
        # {
        #     "epoch": int,
        #     "metrics": dict,
        #     "hyperparameters": dict,
        #     "model_state_dict": dict,
        #     "optimizer_state_dict": dict,
        #     "scheduler_state_dict": dict,
        #     "scaler_state_dict": dict
        # }
        # However, we currently also support checkpoints coming from PyTorch Lightning.
        # In that case, the checkpoint holds the key "pytorch-lightning_version". If the
        # key is present, we assume the checkpoint is coming from PyTorch Lightning.

        is_pl_checkpoint = "pytorch-lightning_version" in checkpoint.keys()
        if is_pl_checkpoint:
            # PyTorch Lightning checkpoint
            hyper_parameters = checkpoint["hyper_parameters"]

            input_size = hyper_parameters['input_size']
            output_size = hyper_parameters['output_size']
            lr = hyper_parameters['lr'] if lr is None else lr
            n_head = hyper_parameters['n_head']
            n_layers = hyper_parameters['n_layers']
            loss = hyper_parameters['loss']

            # Create and load the model
            model = TransformerOracle(
                input_size, output_size, n_head, n_layers, lr, loss)
            model.load_state_dict(checkpoint["state_dict"])

            optimizer_state_dict = checkpoint["optimizer_states"][0]
            scheduler_state_dict = checkpoint["lr_schedulers"][0]

        else:
            # Checkpoint based on the syntax of TransformerOracle.pack_for_checkpoint().
            hyper_parameters = checkpoint["hyperparameters"]

            input_size = hyper_parameters['input_size']
            output_size = hyper_parameters['output_size']
            lr = hyper_parameters['lr'] if lr is None else lr
            n_head = hyper_parameters['n_head']
            n_layers = hyper_parameters['n_layers']
            loss = hyper_parameters['loss']

            # Create and load the model
            model = TransformerOracle(
                input_size, output_size, n_head, n_layers, lr, loss)
            model.load_state_dict(checkpoint["model_state_dict"])

            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            scheduler_state_dict = checkpoint["scheduler_state_dict"]

        # Prepare the checkpoint state dicts for when calling
        # configure_optimizers.
        model.checkpoint_state_dicts = {
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict
        }
        # add the scaler state dict if it exists.
        if "scaler_state_dict" in checkpoint.keys() and checkpoint["scaler_state_dict"] is not None:
            model.checkpoint_state_dicts["scaler_state_dict"] = checkpoint["scaler_state_dict"]

        return model

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_amp):
            y_hat = self(x)
            loss = self.loss(y_hat, y)

        y_int = torch.round(y)

        with torch.no_grad():
            # Compute & log the metrics
            info = {
                'train_loss':       loss.detach(),
                'train_acc':        self.accuracy(y_hat, y_int),
                'train_recall':     self.recall(y_hat, y_int),
                'train_spec':       self.spec(y_hat, y_int),
                'train_precision':  self.precision(y_hat, y_int),
                'train_mse':        self.mse(y_hat, y),
                'train_mae':        self.mae(y_hat, y),
                'train_f1':         self.f1(y_hat, y)
            }

        matrix = {
            'train_positives':  y_int.sum(),
            'train_negatives':  (1 - y_int).sum()
        }

        return loss, info, matrix

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_amp):
                y_hat = self(x)
                loss = self.loss(y_hat, y)
                y_int = torch.round(y)

            # Compute & log the metrics
            info = {
                'val_loss':      loss,
                'val_acc':       self.accuracy(y_hat, y_int),
                'val_recall':    self.recall(y_hat, y_int),
                'val_spec':      self.spec(y_hat, y_int),
                'val_precision': self.precision(y_hat, y_int),
                'val_mse':       self.mse(y_hat, y),
                'val_mae':       self.mae(y_hat, y),
                'val_f1':        self.f1(y_hat, y),
            }

        matrix = {
            'val_positives': y_int.sum(),
            'val_negatives': (1 - y_int).sum(),
            'TP': (y_int * y_hat).sum(),
            'FP': ((1 - y_int) * y_hat).sum(),
            'TN': ((1 - y_int) * (1 - y_hat)).sum(),
            'FN': (y_int * (1 - y_hat)).sum(),
        }

        return loss, info, matrix

    def test_step(self, test_batch, batch_idx, histogram_metrics: dict = None):
        x, y = test_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_amp):
                y_hat = self(x)
                loss = self.loss(y_hat, y)
                y_int = torch.round(y)

            # Compute & log the metrics
            info = {
                'test_loss':      loss,
                'test_acc':       self.accuracy(y_hat, y_int),
                'test_recall':    self.recall(y_hat, y_int),
                'test_spec':      self.spec(y_hat, y_int),
                'test_precision': self.precision(y_hat, y_int),
                'test_mse':       self.mse(y_hat, y),
                'test_mae':       self.mae(y_hat, y),
                'test_f1':        self.f1(y_hat, y),
            }
        # self.roc.update(y_hat, y_int.int())

        if histogram_metrics is not None:
            # Compute histogram bin metrics
            # We want to compute the histogram of the lengths of the tracks as the X axis
            # and the accuracy of the model for each length of streamline as the Y axis.
            # We will use the histogram of the lengths of the tracks to compute the bin
            # metrics.
            # [0, 5[, [5, 10[, [5, 10[, [10, 15[, [15, 20[, [20, 25[, [25, 30[, [30, 35[,
            # [35, 40[, [40, 45[, [45, 50[, [50, 55[, [55, 60[, [60, 65[, [65, 70[,
            # [70, 75[, [75, 80[, [80, 85[, [85, 90[, [90, 95[, [95, 100[, [100, 105[,
            # [105, 110[, [110, 115[, [115, 120[, [120, 125[, [125, 130[, [130, 135[,
            # [135, 140[, [140, 145[, [145, 150[, [150, 155[, [155, 160[, [160, 165[,
            # [165, 170[, [170, 175[, [175, 180[, [180, 185[, [185, 190[, [190, 195[,
            # [195, 200[

            # Get the lengths of the tracks
            from dipy.io.stateful_tractogram import StatefulTractogram, Space
            reference = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz"
            # reference = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz"
            # reference = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_1mm/scoring_data/t1.nii.gz"
            sft = StatefulTractogram(x.cpu().numpy(), reference, Space.VOX)
            sft.to_rasmm()
            lengths = np.asarray(list(length(sft.streamlines)))

            # Compute the histogram of the lengths of the tracks
            bins = np.arange(0, 200, 5).astype(np.float32)

            # Compute the bin metrics
            # "bin_0": (nb_tracks, total_accuracy), "bin_1": (nb_tracks, total_accuracy), ...
            for i in range(len(bins) - 1):
                # Get the indices of the tracks that have a length in the current bin
                indices = np.where((lengths >= bins[i]) & (
                    lengths < bins[i + 1]))[0]

                bin_name = '{:.0f}'.format(bins[i + 1])
                if not bin_name in histogram_metrics.keys():
                    histogram_metrics[bin_name] = defaultdict(int)
                # Get the accuracy of the model for the tracks in the current bin
                if indices.size == 0:
                    # Doing this will initialize the values to zero if it wasn't already
                    histogram_metrics[bin_name]['nb_streamlines'] += 0
                    histogram_metrics[bin_name]['nb_positive'] += 0
                    histogram_metrics[bin_name]['nb_negative'] += 0
                    histogram_metrics[bin_name]['nb_correct'] += 0
                    histogram_metrics[bin_name]['nb_correct_positives'] += 0
                    histogram_metrics[bin_name]['nb_correct_negatives'] += 0
                else:
                    preds = (y_hat[indices] > 0.5).int()
                    gt = y_int[indices]
                    nb_corrects = self.accuracy(
                        y_hat[indices], y_int[indices]).item() * indices.size

                    # Compute the number of positive streamlines that were correctly classified
                    nb_positive = gt.sum().item()
                    nb_negative = (1 - gt).sum().item()
                    nb_correct_positives = (gt * preds).sum().item()
                    nb_correct_negatives = (
                        (1 - gt) * (1 - preds)).sum().item()

                    histogram_metrics[bin_name]['nb_streamlines'] += indices.size
                    histogram_metrics[bin_name]['nb_positive'] += nb_positive
                    histogram_metrics[bin_name]['nb_negative'] += nb_negative
                    histogram_metrics[bin_name]['nb_correct'] += nb_corrects
                    histogram_metrics[bin_name]['nb_correct_positives'] += nb_correct_positives
                    histogram_metrics[bin_name]['nb_correct_negatives'] += nb_correct_negatives

        return loss, info

    def pack_for_checkpoint(self, epoch, metrics, optimizer, scheduler, scaler):
        return {
            'epoch': epoch,
            'metrics': metrics,
            'hyperparameters': {
                'name': self.__class__.__name__,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'n_head': self.n_head,
                'n_layers': self.n_layers,
                'lr': self.lr,
                'loss': self.loss.__class__
            },
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
