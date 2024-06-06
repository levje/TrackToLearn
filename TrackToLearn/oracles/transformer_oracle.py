import math
import torch

import lightning as L
from torch import nn, Tensor
from torchmetrics.regression import (MeanSquaredError, MeanAbsoluteError)
from torchmetrics.classification import (BinaryRecall, BinaryPrecision, BinaryAccuracy, BinaryROC,
                                         BinarySpecificity, BinaryF1Score)

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


class TransformerOracle(L.LightningModule):

    def __init__(
            self,
            input_size,
            output_size,
            n_head,
            n_layers,
            lr,
            loss=nn.MSELoss
    ):
        super(TransformerOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.n_head = n_head
        self.n_layers = n_layers

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
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
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
    def load_from_checkpoint(cls, checkpoint: dict):

        hyper_parameters = checkpoint["hyper_parameters"]

        input_size = hyper_parameters['input_size']
        output_size = hyper_parameters['output_size']
        lr = hyper_parameters['lr']
        n_head = hyper_parameters['n_head']
        n_layers = hyper_parameters['n_layers']
        loss = hyper_parameters['loss']

        model = TransformerOracle(
            input_size, output_size, n_head, n_layers, lr, loss)

        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_int = torch.round(y)

        # Compute & log the metrics
        self.log('train_loss',      loss, on_step=True, on_epoch=False)
        self.log('train_acc',       self.accuracy(y_hat, y_int), on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_recall',    self.recall(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('train_spec',      self.spec(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('train_precision', self.precision(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('train_mse',       self.mse(y_hat, y), on_step=True, on_epoch=False)
        self.log('train_mae',       self.mae(y_hat, y), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_int = torch.round(y)

        # Compute & log the metrics
        self.log('val_loss',      loss, on_step=True, on_epoch=False)
        self.log('val_acc',       self.accuracy(y_hat, y_int), on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_recall',    self.recall(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('val_spec',      self.spec(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('val_precision', self.precision(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('val_mse',       self.mse(y_hat, y), on_step=True, on_epoch=False)
        self.log('val_mae',       self.mae(y_hat, y), on_step=True, on_epoch=False)
        self.log('val_f1',        self.f1(y_hat, y), on_step=True, on_epoch=False)       

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        if len(x.shape) > 3:
            x, y = x.squeeze(0), y.squeeze(0)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_int = torch.round(y)

        # Compute & log the metrics
        self.log('test_loss',      loss, on_step=True, on_epoch=False)
        self.log('test_acc',       self.accuracy(y_hat, y_int), on_step=True, on_epoch=False, prog_bar=True)
        self.log('test_recall',    self.recall(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('test_spec',      self.spec(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('test_precision', self.precision(y_hat, y_int), on_step=True, on_epoch=False)
        self.log('test_mse',       self.mse(y_hat, y), on_step=True, on_epoch=False)
        self.log('test_mae',       self.mae(y_hat, y), on_step=True, on_epoch=False)
        self.log('test_f1',        self.f1(y_hat, y), on_step=True, on_epoch=False)
        self.roc.update(y_hat, y_int.int())

    def on_test_epoch_end(self):
        # TODO: Not implemented.
        return super().on_test_epoch_end()
