import torch
import torch.nn as nn
from TrackToLearn.oracles.transformer_oracle import LightningLikeModule
from TrackToLearn.trainers.oracle.oracle_monitor import OracleMonitor
from TrackToLearn.utils.torch_utils import get_device
from TrackToLearn.algorithms.shared.utils import add_item_to_means, mean_losses
from enum import Enum
from collections import defaultdict
from tqdm import tqdm

def to_device(obj, device):
    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]
    return obj.to(device)

class HookEvent(Enum):
    ON_TRAIN_EPOCH_START = 'on_train_epoch_start'
    ON_TRAIN_EPOCH_END = 'on_train_epoch_end'
    ON_TRAIN_BATCH_START = 'on_train_batch_start'
    ON_TRAIN_BATCH_END = 'on_train_batch_end'
    ON_VAL_BATCH_START = 'on_val_batch_start'
    ON_VAL_BATCH_END = 'on_val_batch_end'
    ON_TEST_START = 'on_test_start'
    ON_TEST_END = 'on_test_end'

class HooksManager(object):
    def __init__(self) -> None:
        self._hooks = {event: [] for event in HookEvent}

    def register_hook(self, event, hook):
        self._hooks[event].append(hook)

    def unregister_hook(self, event, hook):
        self._hooks[event].remove(hook)

    def trigger_hooks(self, event, *args, **kwargs):
        for hook in self._hooks[event]:
            hook(*args, **kwargs)

class OracleTrainer(object):
    def __init__(self,
        experiment,
        experiment_id,
        saving_path,
        max_epochs,
        use_comet=True,
        enable_checkpointing=True,
        val_interval=1,
        device=get_device()
    ):
        self.experiment = experiment
        self.experiment_id = experiment_id
        self.saving_path = saving_path

        self.checkpointing_enabled = enable_checkpointing
        self.device = device
        self.max_epochs = max_epochs
        self.val_interval = val_interval

        self.hooks_manager = HooksManager()
        self.oracle_monitor = OracleMonitor(
            experiment=self.experiment,
            experiment_id=self.experiment_id,
            use_comet=use_comet
        )

    def setup_model_training(self, oracle_model: LightningLikeModule):
        """
        This method must be called before calling fit_iter().

        It is used to configure the optimizer, the scheduler and the scaler.
        Contrary to the standard fit() method from Lightning AI that takes the model
        as an argument, we want to be able to call fit() multiple times with a coherent
        configuration of the optimizer, the scheduler and the scaler to train the same
        model.
        """
        self.oracle_model = oracle_model
        self._reset_optimizers()

    def _verify_model_was_setup(self):
        if not hasattr(self, 'oracle_model') or self.oracle_model is None:
            raise ValueError("You must call setup_model_training before calling fit_iter.\n"
                             "This makes sure the model is properly setup for training, by \n"
                             "configuring the optimizer, the scheduler and the scaler.")
        
    def _reset_optimizers(self):
        optim_info = self.oracle_model.configure_optimizers(self)
        self.optimizer = optim_info['optimizer']
        self.scheduler = optim_info['lr_scheduler']['scheduler']
        self.scaler = optim_info['scaler']

    def fit_iter(
        self,
        train_dataloader,
        val_dataloader,
        reset_optimizers=False
    ):
        """
        This method trains the model for a given number of epochs.
        Contrary to the standard fit() method, this method is made
        to be called multiple times with the same model. This is
        especially useful when training a model iteratively.

        Args:
            train_dataloader: The training dataloader
            val_dataloader: The validation dataloader
            reset_optimizers: If True, the optimizer, the scheduler and the scaler
                are reset before training the model instead of reusing the same
                optimizer, scheduler and scaler as well as their last respective
                states.
        """
        self._verify_model_was_setup()

        self.oracle_model.train() # Set model to training mode
        self.oracle_model = self.oracle_model.to(self.device)

        if reset_optimizers:
            self._reset_optimizers()

        best_acc = 0
        with tqdm(range(len(train_dataloader))) as pbar:
            for epoch in range(self.max_epochs):
                pbar.set_description(f"Training oracle epoch {epoch}")
                self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_EPOCH_START)

                train_metrics = defaultdict(list)
                for i, batch in enumerate(train_dataloader):
                    self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_BATCH_START)

                    batch = to_device(batch, self.device)

                    # Train step
                    loss, train_info = self.oracle_model.training_step(batch, i)
                    add_item_to_means(train_metrics, train_info)
                    train_metrics['lr'].append(torch.tensor(self.optimizer.param_groups[0]['lr']))

                    # Clear gradients
                    self.optimizer.zero_grad()

                    # Backward pass
                    self.scaler.scale(loss).backward()

                    # Update parameters
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    pbar.update()
                    # pbar.set_postfix(train_loss=train_metrics['train_loss'])
                    self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_BATCH_END)
                train_metrics = mean_losses(train_metrics)

                self.oracle_monitor.log_metrics(train_metrics, epoch)

                if epoch % self.val_interval == 0:
                    val_metrics = defaultdict(list)
                    for i, batch in enumerate(val_dataloader):
                        batch = to_device(batch, self.device)
                        self.hooks_manager.trigger_hooks(HookEvent.ON_VAL_BATCH_START)

                        # TODO: Implement validation step
                        _, val_info = self.oracle_model.validation_step(batch, i)
                        add_item_to_means(val_metrics, val_info)

                        self.hooks_manager.trigger_hooks(HookEvent.ON_VAL_BATCH_END)
                    
                    val_metrics = mean_losses(val_metrics)
                    self.oracle_monitor.log_metrics(val_metrics, epoch)

                    # Checkpointing
                    if self.checkpointing_enabled:
                        checkpoint_dict = {
                            'epoch': epoch,
                            'metrics': val_metrics,
                            'model_state_dict': self.oracle_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()
                        }

                        # Always have a copy of the latest model
                        latest_name = '{}/latest_epoch.ckpt'.format(self.saving_path, epoch)
                        torch.save(checkpoint_dict, latest_name)

                        # If the VC is the best so far, save the model with the name best_acc_epoch_{epoch}.ckpt
                        # Also save the optimizer state and the scheduler state, the epoch and the metrics
                        acc = val_metrics['val_acc']
                        if acc > best_acc:
                            best_name = '{}/best_vc_epoch.ckpt'.format(self.saving_path, epoch)
                            torch.save(checkpoint_dict, best_name)
                            best_acc = best_acc

                pbar.reset()
                self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_EPOCH_END)

        self.oracle_model = self.oracle_model.to('cpu')


    def test(self, test_dataloader):
        self.hooks_manager.trigger_hooks(HookEvent.ON_TEST_START)

        self.oracle_model.eval() # Set model to evaluation mode
        self.oracle_model.to(self.device)

        test_metrics = defaultdict(list)
        for i, batch in enumerate(test_dataloader):
            batch = to_device(batch, self.device)
            _, test_info = self.oracle_model.test_step(batch, i)
            add_item_to_means(test_metrics, test_info)

        test_metrics = mean_losses(test_metrics)
        self.hooks_manager.trigger_hooks(HookEvent.ON_TEST_END)
        return test_metrics
