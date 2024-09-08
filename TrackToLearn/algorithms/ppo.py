import numpy as np
import torch

from collections import defaultdict
from torch import nn
from typing import Tuple
from dataclasses import dataclass, field
from copy import deepcopy
from abc import abstractmethod

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.onpolicy import PPOActorCritic, HybridMaxEntropyActor, Actor
from TrackToLearn.algorithms.shared.replay import OnPolicyReplayBuffer, RlhfReplayBuffer
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.utils import fix_streamlines_length
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, old_mean_losses)
from TrackToLearn.utils.utils import TTLProfiler, break_if_found_nans, break_if_params_have_nans, break_if_grads_have_nans
from time import time
from TrackToLearn.algorithms.shared.kl import AdaptiveKLController, FixedKLController

@dataclass
class PPOHParams(object):
    algorithm: str = field(repr=True, init=False, default="PPO")

    reward_function_weighting: float = 10.0

    action_std: float = 0.0
    lr: float = 1.5e-6
    gamma: float = 0.99
    lmbda: float = 0.0
    K_epochs: int = 5
    eps_clip: float = 0.2
    val_clip_coef: float = 0.2
    entropy_loss_coeff: float = 0.0001

    adaptive_kl: bool = False
    kl_penalty_coeff: float = 0.02
    kl_target: float = 0.005
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
        critic_checkpoint: dict = None,
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

        def _assert_same_weights(model1, model2):
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.all(torch.eq(p1, p2))

        # Initialize policy and reference (old) policy.
        self.agent = PPOActorCritic(
            input_size,
            action_size,
            hidden_dims,
            device,
            action_std,
            actor_cls=Actor,
            critic_checkpoint=critic_checkpoint
        ).to(device)
        self.old_agent = deepcopy(self.agent.actor)
        _assert_same_weights(self.agent.actor, self.old_agent)

        def _post_state_dict_agent_hook(module, incompatible_keys):
            """
            Since we are initializing the current and the reference policy
            with the same weights, we need to make sure that when there's a
            checkpoint loaded for the current policy (initially), the reference
            should also be updated with the same weights.
            """
            self.old_agent.load_state_dict(self.agent.actor.state_dict())
            
        # The old agent should also be initialized to the loaded checkpoint.
        self.agent.actor.register_load_state_dict_post_hook(_post_state_dict_agent_hook)

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
        self.streamline_nb_points = 128
        self.verbose = False

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
        s, st, a, ret, adv, p, val, *_ = replay_buffer.sample()

        # PPO allows for multiple gradient steps on the same data
        # TODO: Should be switched with the batch ?
        for _ in range(self.K_epochs):
            pr = TTLProfiler(enabled=False, throw_at_stop=True)
            pr.start()
            for i in range(0, len(s), batch_size):
                # you can slice further than an array's length
                j = i + batch_size
                
                streamline = st[i:j]
                state = s[i:j]
                action = a[i:j]
                old_prob = p[i:j]
                returns = ret[i:j]
                advantage = adv[i:j]
                old_vals = val[i:j]

                # V_pi'(s) and pi'(a|s)
                v, logprob, entropy, *_ = self.agent.evaluate(
                    state,
                    action,
                    probabilistic=1.0,
                    streamlines=streamline)

                # Ratio between probabilities of action according to policy and
                # target policies
                assert logprob.size() == old_prob.size(), \
                    '{}, {}'.format(logprob.size(), old_prob.size())
                ratio = torch.exp(logprob - old_prob)
                if (torch.abs(ratio) > 1e10).any():
                    indexes = (torch.abs(ratio) > 1e10).nonzero()
                    breakpoint()
                    print("Ratio is enormous")

                break_if_found_nans(logprob)
                break_if_found_nans(old_prob)
                break_if_found_nans(ratio)

                # Surrogate policy loss
                assert ratio.size() == advantage.size(), \
                    '{}, {}'.format(ratio.size(), advantage.size())

                # Finding V Loss:
                assert returns.size() == v.size(), \
                    '{}, {}'.format(returns.size(), v.size())

                break_if_found_nans(advantage)
                surrogate_policy_loss_1 = ratio * advantage
                break_if_found_nans(surrogate_policy_loss_1) # REMOVE
                surrogate_policy_loss_2 = torch.clamp(
                    ratio,
                    1-self.eps_clip,
                    1+self.eps_clip) * advantage
                
                if (torch.abs(surrogate_policy_loss_1) > 1e10).any():
                    indexes = (torch.abs(surrogate_policy_loss_1) > 1e10).nonzero()
                    breakpoint()
                    print("Loss1 is enormous")
                if (torch.abs(surrogate_policy_loss_2) > 1e10).any():
                    indexes = (torch.abs(surrogate_policy_loss_2) > 1e10).nonzero()
                    breakpoint()
                    print("Loss2 is enormous")

                break_if_found_nans(surrogate_policy_loss_2) # REMOVE

                # PPO "pessimistic" policy loss
                actor_loss = -(torch.min(
                    surrogate_policy_loss_1,
                    surrogate_policy_loss_2)).mean() + \
                    -self.entropy_loss_coeff * entropy.mean()
                
                break_if_found_nans(actor_loss) # REMOVE

                # Clip value as in PPO's implementation:
                # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75
                # v_clipped = old_vals + torch.clamp(v - old_vals, -self.val_clip_coef, self.val_clip_coef)
                # critic_loss_clipped = ((v_clipped - returns) ** 2).mean()

                # AC Critic loss
                critic_loss_unclipped = ((v - returns) ** 2).mean()

                # critic_loss = torch.mean(torch.maximum(critic_loss_clipped, critic_loss_unclipped))
                critic_loss = critic_loss_unclipped

                self.optimizer.zero_grad()

                break_if_found_nans(critic_loss) # REMOVE
                break_if_found_nans(actor_loss) # REMOVE

                ((critic_loss * 0.5) + actor_loss).backward()

                break_if_grads_have_nans(self.agent.named_parameters())

                # Gradient step
                gradients = [param.grad.detach().flatten() for param in self.agent.parameters() if param.grad is not None]
                nn.utils.clip_grad_norm_(self.agent.parameters(),
                                         0.5)
                clipped_gradients = [param.grad.detach().flatten() for param in self.agent.parameters() if param.grad is not None]
                gradients_norm = torch.cat(gradients).norm()
                clipped_gradients_norm = torch.cat(clipped_gradients).norm()

                def get_losses():
                    def get_critic_loss():
                        return critic_loss.item()
                    def get_actor_loss():
                        return actor_loss.item()
                    def get_ratio():
                        return ratio.mean().item()
                    def get_surrogate_loss_1():
                        return surrogate_policy_loss_1.mean().item()
                    def get_surrogate_loss_2():
                        return surrogate_policy_loss_2.mean().item()
                    def get_advantage():
                        return advantage.mean().item()
                    def get_entropy():
                        return entropy.mean().item()
                    def get_ret():
                        return returns.mean().item()
                    def get_v():
                        return v.mean().item()

                    losses = {
                        'ratio': get_ratio(),
                        'critic_loss': get_critic_loss(),
                        'actor_loss': get_actor_loss(),
                        'surrogate_loss_1': get_surrogate_loss_1(),
                        'surrogate_loss_2': get_surrogate_loss_2(),
                        'advantage': get_advantage(),
                        'entropy': get_entropy(),
                        'ret': get_ret(),
                        'v': get_v(),
                        'gradients_norm': gradients_norm.item(),
                        'clipped_gradients_norm': clipped_gradients_norm.item(),
                        'kl_penalty': self.kl_penalty_ctrler.value
                    }
                    return losses
                
                losses = get_losses()
                running_losses = add_item_to_means(running_losses, losses)

                self.optimizer.step()

                # check if parameters of the policy become nan
                break_if_params_have_nans(self.agent.parameters())

            pr.stop()

        return old_mean_losses(running_losses)
    
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
        running_mean_ratio = 0
        state = initial_state
        streamlines = np.zeros(
            (state.shape[0], self.max_traj_length, 3),
            dtype=np.float64)
        done = False
        running_losses = defaultdict(list)
        running_reward_factors = defaultdict(list)

        episode_length = 0
        indices = np.asarray(range(state.shape[0]))

        while not np.all(done):
            if self.verbose:
                print('_episode ({}) ...\r'.format(episode_length), end='')

            # Select action according to policy
            # Noise is already added by the policy
            break_if_found_nans(state) # REMOVE
            with torch.no_grad():
                action = self.agent.select_action(
                    state, probabilistic=1.0)

            self.t += action.shape[0]

            resampled_streamlines = \
                fix_streamlines_length(streamlines, episode_length, self.streamline_nb_points)
            
            v, cur_logprobs, _ = self.agent.get_evaluation(
                state,
                action,
                probabilistic=1.0,
                streamlines=resampled_streamlines)

            # Perform action
            next_state, next_streamlines, reward, done, info = env.step(action.to(device='cpu', copy=True).numpy())
            break_if_found_nans(next_state) # REMOVE
            reward = torch.as_tensor(reward, dtype=torch.float32)
            running_reward_factors = add_item_to_means(
                running_reward_factors, info['reward_info'])

            resampled_next_streamlines = \
                fix_streamlines_length(next_streamlines, episode_length, self.streamline_nb_points)

            vp, *_ = self.agent.get_evaluation(
                next_state,
                action,
                probabilistic=1.0,
                streamlines=resampled_next_streamlines)

            with torch.no_grad():
                ref_pi = self.old_agent.forward(state, probabilistic=1.0)
                ref_logprobs = ref_pi.log_prob(action).sum(axis=-1)
                
                log_ratio = (cur_logprobs - ref_logprobs).to('cpu')
                running_mean_ratio += log_ratio.mean().item()
                total_reward = reward - self.kl_penalty_ctrler.value \
                    * self.hparams.reward_function_weighting * log_ratio
            self.kl_penalty_ctrler.update(log_ratio, episode_length)

            # Set next state as current state
            running_reward += sum(total_reward)

            # Store data in replay buffer
            with torch.no_grad():
                rb_device = self.replay_buffer.storing_device
                self.replay_buffer.add(
                    torch.tensor(indices),
                    state.to(device=rb_device, copy=True),
                    torch.as_tensor(resampled_streamlines, device=rb_device, dtype=torch.float32),
                    action.to(device=rb_device, copy=True),
                    next_state.to(device=rb_device, copy=True),
                    torch.as_tensor(total_reward, device=rb_device, dtype=torch.float32),
                    torch.as_tensor(done, device=rb_device, dtype=torch.float32),
                    v.to(device=rb_device, copy=True),
                    vp.to(device=rb_device, copy=True),
                    cur_logprobs.to(device=rb_device, copy=True))

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            state, streamlines, idx = env.harvest()
            break_if_found_nans(state) # REMOVE

            indices = indices[idx]

            # Keeping track of episode length
            episode_length += 1

        if self.verbose:
            print("_update...", end='')
        
        losses = self.update(
            self.replay_buffer)
        
        if self.verbose:
            print("done")

        running_losses = add_item_to_means(running_losses, losses)
        running_mean_ratio /= episode_length

        return (
            running_reward,
            running_losses,
            episode_length,
            running_reward_factors,
            running_mean_ratio
            )
