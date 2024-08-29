import torch
import torch.nn as nn
from TrackToLearn.oracles.transformer_oracle import LightningLikeModule
from TrackToLearn.trainers.oracle.oracle_monitor import OracleMonitor
from TrackToLearn.utils.torch_utils import get_device
from TrackToLearn.algorithms.shared.utils import add_item_to_means, mean_losses
from enum import Enum
from collections import defaultdict

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

    def fit_iter(
        self,
        oracle_model: LightningLikeModule,
        train_dataloader,
        val_dataloader
    ):
        
        # TODO: Implement learning rate monitoring
        # TODO: Implement checkpointing
        # TODO: Implement logging
        oracle_model.train() # Set model to training mode

        optim_info = oracle_model.configure_optimizers(self)
        # oracle_model = oracle_model.to(self.device)

        optimizer = optim_info['optimizer']
        scheduler = optim_info['lr_scheduler']['scheduler']

        best_VC = 0
        for epoch in range(self.max_epochs):
            self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_EPOCH_START)

            train_metrics = {}
            for i, batch in enumerate(train_dataloader):
                batch = to_device(batch, self.device)

                # Call hooks _on_train_batch_start
                self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_BATCH_START)

                # Train step
                loss, train_info = oracle_model.training_step(batch, i)
                train_metrics = add_item_to_means(train_metrics, train_info)
                train_metrics['lr'] = optimizer.param_groups[0]['lr']

                # Clear gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()
                scheduler.step()

                self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_BATCH_END)
            train_metrics = mean_losses(train_metrics)

            self.oracle_monitor.log_metrics(train_metrics, epoch)

            if epoch % self.val_interval == 0:
                val_metrics = {}
                for i, batch in enumerate(val_dataloader):
                    batch = to_device(batch, self.device)
                    self.hooks_manager.trigger_hooks(HookEvent.ON_VAL_BATCH_START)

                    # TODO: Implement validation step
                    _, val_info = oracle_model.validation_step(batch, i)
                    val_metrics = add_item_to_means(val_metrics, val_info)

                    self.hooks_manager.trigger_hooks(HookEvent.ON_VAL_BATCH_END)
                
                val_metrics = mean_losses(val_metrics)
                self.oracle_monitor.log_metrics(val_metrics, epoch)

                # Checkpointing
                if self.checkpointing_enabled:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'metrics': val_metrics,
                        'model_state_dict': oracle_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }

                    # Always have a copy of the latest model
                    latest_name = '{}/latest_epoch.ckpt'.format(self.saving_path, epoch)
                    torch.save(checkpoint_dict, latest_name)

                    # If the VC is the best so far, save the model with the name best_vc_epoch_{epoch}.ckpt
                    # Also save the optimizer state and the scheduler state, the epoch and the metrics
                    VC = val_metrics['VC']
                    if VC > best_VC:
                        best_name = '{}/best_vc_epoch.ckpt'.format(self.saving_path, epoch)
                        torch.save(checkpoint_dict, best_name)
                        best_VC = VC

            self.hooks_manager.trigger_hooks(HookEvent.ON_TRAIN_EPOCH_END)

        oracle_model.to('cpu')


    def test(self, oracle_model, test_dataloader):
        self.hooks_manager.trigger_hooks(HookEvent.ON_TEST_START)

        oracle_model.eval() # Set model to evaluation mode
        oracle_model.to(self.device)

        test_metrics = {}
        for i, batch in enumerate(test_dataloader):
            _, test_info = oracle_model.test_step(batch, i)
            test_metrics = add_item_to_means(test_metrics, test_info)

        test_metrics = mean_losses(test_metrics)
        self.hooks_manager.trigger_hooks(HookEvent.ON_TEST_END)
        return test_metrics
