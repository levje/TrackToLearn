import copy
import numpy as np
import torch

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from TrackToLearn.algorithms.sac import SAC
from TrackToLearn.algorithms.shared.offpolicy import SACActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer
from TrackToLearn.utils.torch_utils import get_device
from TrackToLearn.algorithms.shared.kl import AdaptiveKLController, FixedKLController

LOG_STD_MAX = 2
LOG_STD_MIN = -20

@dataclass
class SACAutoHParams:
    lr: float = 3e-4
    gamma: float = 0.99
    n_actors: int = 4096

    alpha: float = 0.2
    batch_size: int = 2**12
    replay_size: int = 1e6

    adaptive_kl: bool = False
    kl_penalty_coeff: float = 0.02
    kl_target: float = 0.005
    kl_horizon: int = 1000

class SACAuto(SAC):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ...
        & Levine, S. (2018). Soft actor-critic algorithms and applications.
        arXiv preprint arXiv:1812.05905.

    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: int,
        hparams: SACAutoHParams = SACAutoHParams(),
        rng: np.random.RandomState = None,
        device: torch.device = get_device,
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Dimensions of the hidden layers
        lr: float
            Learning rate for the optimizer(s)
        gamma: float
            Discount factor
        alpha: float
            Initial entropy coefficient (temperature).
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size to sample the memory
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to use for the algorithm. Should be either "cuda:0"
        """
        self.hparams = hparams
        
        # TO REMOVE
        self.batch_size = hparams.batch_size
        self.gamma = hparams.gamma
        self.alpha = hparams.alpha
        self.n_actors = hparams.n_actors
        self.replay_size = hparams.replay_size

        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.device = device

        self.rng = rng

        def _assert_same_weights(model1, model2):
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.all(torch.eq(p1, p2))

        # Initialize main agent
        self.agent = SACActorCritic(
            input_size, action_size, hidden_dims, device,
        )
        self.old_agent = copy.deepcopy(self.agent.actor)
        _assert_same_weights(self.agent.actor, self.old_agent)

        def _post_state_dict_agent_hook(module, incompatible_keys):
            """
            Since we are initializing the current and the reference policy
            with the same weights, we need to make sure that when there's a
            checkpoint loaded for the current policy (initially), the reference
            should also be updated with the same weights.
            """
            self.old_agent.load_state_dict(self.agent.actor.state_dict())

        self.agent.actor.register_load_state_dict_post_hook(_post_state_dict_agent_hook)

        # Auto-temperature adjustment
        # SAC automatically adjusts the temperature to maximize entropy and
        # thus exploration, but reduces it over time to converge to a
        # somewhat deterministic policy.
        starting_temperature = np.log(self.hparams.alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)
        # Optimizer for alpha
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.hparams.lr)

        # Initialize target agent to provide baseline
        self.target = copy.deepcopy(self.agent)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=self.hparams.lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=self.hparams.lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_agent = False

        self.start_timesteps = 80000
        self.total_it = 0
        self.tau = 0.005
        self.agent_freq = 1

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size, max_size=self.hparams.replay_size)

        self.rng = rng

        if self.hparams.adaptive_kl:
            self.kl_penalty_ctrler = AdaptiveKLController(
                self.hparams.kl_penalty_coeff, self.hparams.kl_target, self.hparams.kl_horizon)
        else:
            self.kl_penalty_ctrler = FixedKLController(self.hparams.kl_penalty_coeff)

    def load_checkpoint(self, checkpoint_file: str):
        """
        Load a checkpoint into the algorithm.

        Parameters
        ----------
        checkpoint: dict
            Dictionary containing the checkpoint to load.
        """
        checkpoint = torch.load(checkpoint_file)

        self.agent.load_checkpoint(checkpoint['agent'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

    def save_checkpoint(self, checkpoint_file: str):
        """
        Save the current state of the algorithm into a checkpoint.

        Parameters
        ----------
        checkpoint_file: str
            File to save the checkpoint into.
        """
        checkpoint = {
            'agent': self.agent.state_dict(as_dict=True),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }

        torch.save(checkpoint, checkpoint_file)

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC Auto improves upon SAC by automatically adjusting the temperature
        parameter alpha. This is done by optimizing the temperature parameter
        alpha to maximize the entropy of the policy. This is done by
        maximizing the following objective:
            J_alpha = E_pi [log pi(a|s) + alpha H(pi(.|s))]
        where H(pi(.|s)) is the entropy of the policy.


        Parameters
        ----------
        batch: Tuple containing the batch of data to train on.

        Returns
        -------
        losses: dict
            Dictionary containing the losses of the algorithm and various
            other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch
        # Compute \pi_\theta(s_t) and log \pi_\theta(s_t)
        pi, logp_pi = self.agent.act(
            state, probabilistic=1.0)
        # Compute the temperature loss and the temperature
        alpha_loss = -(self.log_alpha * (
            logp_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha.exp()

        # Compute the Q values and the minimum Q value
        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized agent loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        with torch.no_grad():
            # Target actions come from *current* agent
            next_action, logp_next_action = self.agent.act(
                next_state, probabilistic=1.0)

            # Compute the next Q values using the target agent
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute the backup which is the Q-learning "target"
            backup = reward + self.gamma * not_done * \
                (target_Q - alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.agent.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_Q1, backup.detach()).mean()
        loss_q2 = F.mse_loss(current_Q2, backup.detach()).mean()
        # Total critic loss
        critic_loss = loss_q1 + loss_q2

        losses = {
            # 'actor_loss': actor_loss.detach(),
            # 'alpha_loss': alpha_loss.detach(),
            # 'critic_loss': critic_loss.detach(),
            # 'loss_q1': loss_q1.detach(),
            # 'loss_q2': loss_q2.detach(),
            # 'entropy': alpha.detach(),
            # 'Q1': current_Q1.mean().detach(),
            # 'Q2': current_Q2.mean().detach(),
            # 'backup': backup.mean().detach(),
        }

        # Optimize the temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.agent.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.agent.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return losses
