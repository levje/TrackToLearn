import math
import torch

from torch import nn, Tensor
from torchmetrics.regression import (MeanSquaredError, MeanAbsoluteError)
from torchmetrics.classification import (BinaryRecall, BinaryPrecision,
                                         BinaryAccuracy, BinaryROC,
                                         BinarySpecificity, BinaryF1Score)
from dipy.tracking.utils import length
import numpy as np
from collections import defaultdict

from TrackToLearn.oracles.transformer_oracle import LightningLikeModule, _verify_out_activation_with_data


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels *
                      4, kernel_size=7, stride=1, padding=3),
            LayerNorm(channels*4),
            # Layer norm
            nn.GELU(),
            nn.Conv1d(in_channels=channels*4, out_channels=channels,
                      kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self.seq(x)


class CnnOracle(LightningLikeModule):

    def __init__(
            self,
            input_size,
            output_size,
            lr,
            loss=nn.MSELoss,
            mixed_precision=True,
            out_activation=nn.Sigmoid
    ):
        super(CnnOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.enable_amp = mixed_precision
        self.out_activation = out_activation
        self.is_binary_classif = self.out_activation == nn.Sigmoid

        self.nb_dirs = 127

        
        self.seq = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=0),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=0),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.fc_out = nn.Linear(4096, output_size)
        self.out = self.out_activation()

        # Loss function
        self.loss = loss()

        if self.is_binary_classif:
            self.accuracy = BinaryAccuracy()
            self.recall = BinaryRecall()
            self.spec = BinarySpecificity()
            self.precision = BinaryPrecision()
            self.roc = BinaryROC()
            self.f1 = BinaryF1Score()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        # Metrics


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
        x = torch.swapaxes(x, 1, 2)
        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims

        h = self.seq(x)
        h = h.view(N, -1) # Flatten
        h = self.fc_out(h)
        y = self.out(h)

        return y.squeeze()

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

        # Checkpoint based on the syntax of TransformerOracle.pack_for_checkpoint().
        hyper_parameters = checkpoint["hyperparameters"]

        input_size = hyper_parameters['input_size']
        output_size = hyper_parameters['output_size']
        lr = hyper_parameters['lr'] if lr is None else lr
        loss = hyper_parameters['loss']

        if 'output_activation' in hyper_parameters.keys():
            out_activation = hyper_parameters['output_activation']
        else:
            out_activation = nn.Sigmoid

        # Create and load the model
        model = CnnOracle(
            input_size, output_size, lr, loss, True, out_activation)
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

        _verify_out_activation_with_data(self.out_activation, y)

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_amp):
            y_hat = self(x)
            loss = self.loss(y_hat, y)

        y_int = torch.round(y)

        with torch.no_grad():
            # Compute & log the metrics
            info = {
                # 'train_loss':       loss.detach(),
                'train_mse':        self.mse(y_hat, y),
                'train_mae':        self.mae(y_hat, y),
            }

            if self.is_binary_classif:
                info.update({
                    'train_acc':        self.accuracy(y_hat, y_int),
                    'train_recall':     self.recall(y_hat, y_int),
                    'train_spec':       self.spec(y_hat, y_int),
                    'train_precision':  self.precision(y_hat, y_int),
                    'train_f1':         self.f1(y_hat, y)
                })

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
                'val_mse':       self.mse(y_hat, y),
                'val_mae':       self.mae(y_hat, y),
            }

            if self.is_binary_classif:
                info.update({
                    'val_acc':       self.accuracy(y_hat, y_int),
                    'val_recall':    self.recall(y_hat, y_int),
                    'val_spec':      self.spec(y_hat, y_int),
                    'val_precision': self.precision(y_hat, y_int),
                    'val_f1':        self.f1(y_hat, y)
                })

        # Since we have a range of [-1, 1] for the
        # labels. Required for the following lines.
        y_int[y_int == -1] = 0

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
                'test_mse':       self.mse(y_hat, y),
                'test_mae':       self.mae(y_hat, y),
            }

            if self.is_binary_classif:
                info.update({
                    'test_acc':       self.accuracy(y_hat, y_int),
                    'test_recall':    self.recall(y_hat, y_int),
                    'test_spec':      self.spec(y_hat, y_int),
                    'test_precision': self.precision(y_hat, y_int),
                    'test_f1':        self.f1(y_hat, y),
                })
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

    @property
    def hyperparameters(self):
        return {
            'name': self.__class__.__name__,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'lr': self.lr,
            'loss': self.loss.__class__,
            'output_activation': self.out_activation.__class__,
        }

    def pack_for_checkpoint(self, epoch, metrics, optimizer, scheduler, scaler):
        return {
            'epoch': epoch,
            'metrics': metrics,
            'hyperparameters': self.hyperparameters,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
