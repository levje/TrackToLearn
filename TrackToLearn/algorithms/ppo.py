import numpy as np
import torch

from collections import defaultdict
from torch import nn
from typing import Tuple
from dataclasses import dataclass, field

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.onpolicy import PPOActorCritic
from TrackToLearn.algorithms.shared.replay import OnPolicyReplayBuffer, RlhfReplayBuffer
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)


# KL Controllers based on
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L107
class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

@dataclass
class PPOHParams(object):
    algorithm: str = field(repr=True, init=False, default="PPO")

    action_std: float = 0.0
    lr: float = 1.5e-6
    gamma: float = 0.99
    lmbda: float = 0.99
    K_epochs: int = 80
    eps_clip: float = 0.2
    val_clip_coef: float = 0.2
    entropy_loss_coeff: float = 0.01

    adaptive_kl: bool = False
    kl_penalty_coeff: float = 0.05
    kl_target: float = 0.01
    kl_horizon: int = 1000

# TODO : ADD TYPES AND DESCRIPTION
class PPO(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on
        John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford:
            “Proximal Policy Optimization Algorithms”, 2017;
            http://arxiv.org/abs/1707.06347 arXiv:1707.06347

    Implementation is based on
    - https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py # noqa E501
    - https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py
    - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py # noqa E501

    Some alterations have been made to the algorithms so it could be fitted to the
    tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: str,
        hparams: PPOHParams,
        action_std: float = 0.0,
        max_traj_length: int = 1,
        n_actors: int = 4096,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Widths and layers of the NNs
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        entropy_loss_coeff: float
            Entropy bonus for the actor loss
        K_epochs: int
            How many epochs to run the optimizer using the current samples
            PPO allows for many training runs on the same samples
        max_traj_length: int
            Maximum trajectory length to store in memory.
        eps_clip: float
            Clipping parameter for PPO
        entropy_loss_coeff: float,
            Loss coefficient on policy entropy
            Should sum to 1 with other loss coefficients
        n_actors: int
            Number of learners.
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
        """

        self.input_size = input_size
        self.action_size = action_size
        self.hparams = hparams

        self.lr = self.hparams.lr
        self.gamma = self.hparams.gamma

        self.on_policy = True

        # Declare policy
        self.agent = PPOActorCritic(
            input_size, action_size, hidden_dims, device, action_std,
        ).to(device)

        # Note the optimizer is ran on the target network's params
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=self.hparams.lr)

        # GAE Parameter
        self.lmbda = self.hparams.lmbda

        # PPO Specific parameters
        self.max_traj_length = max_traj_length
        self.K_epochs = self.hparams.K_epochs
        self.lmbda = self.hparams.lmbda
        self.eps_clip = self.hparams.eps_clip
        self.val_clip_coef = self.hparams.val_clip_coef
        self.entropy_loss_coeff = self.hparams.entropy_loss_coeff

        if self.hparams.adaptive_kl:
            self.kl_penalty_ctrler = AdaptiveKLController(
                self.hparams.kl_penalty_coeff, self.hparams.kl_target, self.hparams.kl_horizon)
        else:
            self.kl_penalty_ctrler = FixedKLController(self.hparams.kl_penalty_coeff)

        self.max_action = 1.
        self.t = 1
        self.device = device
        self.n_actors = n_actors

        # Replay buffer
        self.replay_buffer = RlhfReplayBuffer(
            input_size, action_size, n_actors,
            max_traj_length, self.gamma, self.lmbda)

        self.rng = rng

    def update(
        self,
        replay_buffer,
        batch_size=4096,
    ) -> Tuple[float, float]:
        """
        Policy update function, where we want to maximize the probability of
        good actions and minimize the probability of bad actions

        The general idea is to compare the current policy and the target
        policies. To do so, the "ratio" is calculated by comparing the
        probabilities of actions for both policies. The ratio is then
        multiplied by the "advantage", which is how better than average
        the policy performs.

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

        PPO adds a twist to this where, since the advantage estimation is done
        with your (potentially bad) networks, a "pessimistic view" is used
        where gains will be clamped, so that high gradients (for very probable
        or with a high-amplitude advantage) are tamed. This is to prevent your
        network from diverging too much in the early stages

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """

        running_losses = defaultdict(list)

        # Sample replay buffer
        # s, a, r, p, val, next_val, l, nd, *_ = \
            # replay_buffer.sample()
        s, a, ret, adv, p, val = replay_buffer.sample()

        # PPO allows for multiple gradient steps on the same data
        # TODO: Should be switched with the batch ?
        for _ in range(self.K_epochs):

            for i in range(0, len(s), batch_size):
                # you can slice further than an array's length
                j = i + batch_size

                state = torch.FloatTensor(s[i:j]).to(self.device)
                action = torch.FloatTensor(a[i:j]).to(self.device)
                old_prob = torch.FloatTensor(p[i:j]).to(self.device)
                old_vals = torch.FloatTensor(val[i:j]).to(self.device)
                old_next_vals = torch.FloatTensor(next_val[i:j]).to(self.device)

                # V_pi'(s) and pi'(a|s)
                v, logprob, entropy, *_ = self.agent.evaluate(
                    state,
                    action,
                    probabilistic=1.0)

                reward = torch.FloatTensor(r[i:j]).to(self.device)
                lengths = torch.LongTensor(l[i:j]).to(self.device)
                not_dones = torch.FloatTensor(nd[i:j]).to(self.device)
                # returns = torch.FloatTensor(ret[i:j]).to(self.device)
                # advantage = torch.FloatTensor(adv[i:j]).to(self.device)

                # Ratio between probabilities of action according to policy and
                # target policies
                assert logprob.size() == old_prob.size(), \
                    '{}, {}'.format(logprob.size(), old_prob.size())
                ratio = torch.exp(logprob - old_prob)
                
                # Apply KL penalty to rewards
                # TODO: The KL penalty should be only applied to the oracle's reward.
                # This should be weighted by the oracle weighting factor in the reward function.
                total_reward = reward - self.kl_penalty_ctrler.value * ratio


                # Returns and advantage calculated with KL-penalty for RLHF
                returns, advantage = self.replay_buffer.compute_adv_rets(total_reward,
                                                                         lengths,
                                                                         old_vals,
                                                                         old_next_vals,
                                                                         not_dones)

                # Surrogate policy loss
                assert ratio.size() == advantage.size(), \
                    '{}, {}'.format(ratio.size(), advantage.size())

                # Finding V Loss:
                assert returns.size() == v.size(), \
                    '{}, {}'.format(returns.size(), v.size())

                surrogate_policy_loss_1 = ratio * advantage
                surrogate_policy_loss_2 = torch.clamp(
                    ratio,
                    1-self.eps_clip,
                    1+self.eps_clip) * advantage

                # PPO "pessimistic" policy loss
                actor_loss = -(torch.min(
                    surrogate_policy_loss_1,
                    surrogate_policy_loss_2)).mean() + \
                    -self.entropy_loss_coeff * entropy.mean()

                # Clip value as in PPO's implementation:
                # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75
                # v_clipped = old_vals + torch.clamp(v - old_vals, -self.val_clip_coef, self.val_clip_coef)

                # AC Critic loss
                critic_loss_unclipped = ((v - returns) ** 2).mean()
                # critic_loss_clipped = ((v_clipped - returns) ** 2).mean()

                # critic_loss = torch.mean(torch.maximum(critic_loss_clipped, critic_loss_unclipped))
                critic_loss = critic_loss_unclipped

                losses = {
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'ratio': ratio.mean().item(),
                    'surrogate_loss_1': surrogate_policy_loss_1.mean().item(),
                    'surrogate_loss_2': surrogate_policy_loss_2.mean().item(),
                    'advantage': advantage.mean().item(),
                    'entropy': entropy.mean().item(),
                    'ret': returns.mean().item(),
                    'v': v.mean().item(),
                }

                running_losses = add_item_to_means(running_losses, losses)

                self.optimizer.zero_grad()
                ((critic_loss * 0.5) + actor_loss).backward()

                # Gradient step
                nn.utils.clip_grad_norm_(self.agent.parameters(),
                                         0.5)
                self.optimizer.step()

        return {}
    
    def _episode(
        self,
        initial_state: np.ndarray,
        env: BaseEnv,
    ) -> Tuple[float, float, float, int]:
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done
        Gather transitions and train on them according to the RL algorithm's
        rules.

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        running_reward: float
            Cummulative training steps reward
        actor_loss: float
            Policty gradient loss of actor
        critic_loss: float
            MSE loss of critic
        episode_length: int
            Length of episode aka how many transitions were gathered
        """

        running_reward = 0
        state = initial_state
        done = False
        running_losses = defaultdict(list)
        running_reward_factors = defaultdict(list)

        episode_length = 0
        indices = np.asarray(range(state.shape[0]))

        while not np.all(done):

            # Select action according to policy
            # Noise is already added by the policy
            with torch.no_grad():
                action = self.agent.select_action(
                    state, probabilistic=1.0)

            self.t += action.shape[0]

            v, prob, _ = self.agent.get_evaluation(
                state,
                action,
                probabilistic=1.0)

            # Perform action
            next_state, reward, done, info = env.step(action.to(device='cpu', copy=True).numpy())
            running_reward_factors = add_item_to_means(
                running_reward_factors, info['reward_info'])

            vp, *_ = self.agent.get_evaluation(
                next_state,
                action,
                probabilistic=1.0)

            # Set next state as current state
            running_reward += sum(reward)

            # Store data in replay buffer
            self.replay_buffer.add(
                indices,
                state.cpu().numpy(),
                action.cpu().numpy(),
                next_state.cpu().numpy(),
                reward,
                done,
                v,
                vp,
                prob)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            state, idx = env.harvest()

            indices = indices[idx]

            # Keeping track of episode length
            episode_length += 1

        losses = self.update(
            self.replay_buffer)
        running_losses = add_item_to_means(running_losses, losses)

        return (
            running_reward,
            running_losses,
            episode_length,
            running_reward_factors)
