
from abc import abstractmethod
import numpy as np

# KL Controllers based on
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L107
class RlhfKLController:
    def __init__(self, kl_coef: float, mode_init: str = 'pretrain', pretrain_kl_coef: float = 0.0):
        self._value = kl_coef
        self._pretrain_kl_coef = pretrain_kl_coef
        if mode_init == 'pretrain':
            self.pretrain = True
        elif mode_init == 'rlhf':
            self.pretrain = False
        else:
            raise ValueError('invalid value for mode_init {}.'
                             'Should be either "pretrain" or "rlhf"'.format(mode_init))

    def pretrain_mode(self):
        self.pretrain = True

    def rlhf_mode(self):
        self.pretrain = False    
    
    @property
    def value(self):
        if self.pretrain:
            # potentially no KL penalty during pretraining
            return self._pretrain_kl_coef
        else:
            return self._value
        
    @value.setter
    def value(self, value):
        self._value = value

    @abstractmethod 
    def update(self, current, n_steps):
        raise NotImplementedError("KL Controller must implement update method")
        
class FixedKLController(RlhfKLController):
    def __init__(self, kl_coef):
        super().__init__(kl_coef)

    def update(self, current, n_steps):
        pass

class AdaptiveKLController(RlhfKLController):
    def __init__(self, init_kl_coef, target, horizon):
        super().__init__(init_kl_coef)
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult