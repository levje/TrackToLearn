import numpy as np
import torch

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)
from TrackToLearn.utils.torch_utils import assert_accelerator, get_device
import scipy.signal

assert_accelerator()

class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        action_std: float = 0.0,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(Actor, self).__init__()

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim, activation=nn.Tanh)

        # State-independent STD, as opposed to SAC which uses a
        # state-dependent STD.
        # See https://spinningup.openai.com/en/latest/algorithms/sac.html
        # in the "You Should Know" box
        log_std = -action_std * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def _mu(self, state: torch.Tensor):
        return self.layers(state)

    def _distribution(self, state: torch.Tensor):
        mu = self._mu(state)
        std = torch.exp(self.log_std)
        try:
            dist = Normal(mu, std)
        except ValueError as e:
            print(mu, std)
            raise e

        return dist

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        return self._distribution(state)


class PolicyGradient(nn.Module):
    """ PolicyGradient module that handles actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        action_std: float = 0.0,
    ):
        super(PolicyGradient, self).__init__()
        self.action_dim = action_dim

        self.actor = Actor(
            state_dim, action_dim, hidden_dims, action_std,
        ).to(get_device())

    def act(
        self, state: torch.Tensor, probabilistic: float = 1.0,
    ) -> torch.Tensor:
        """ Select noisy action according to actor
        """
        pi = self.actor.forward(state)
        # Should always be stochastic
        if probabilistic > 0.0:
            action = pi.sample()  # if stochastic else pi.mean
        else:
            action = pi.mean

        return action

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        pi = self.actor(state)
        mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()

        return action_logprob, entropy, mu, std

    def select_action(
        self, state: np.array, probabilistic=1.0,
    ) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """

        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=get_device())
        action = self.act(state, probabilistic).cpu().data.numpy()

        return action

    def get_evaluation(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=get_device())
        action = torch.as_tensor(action, dtype=torch.float32, device=get_device())

        prob, entropy, mu, std = self.evaluate(state, action)

        # REINFORCE does not use a critic
        values = np.zeros((state.size()[0]))

        return (
            values,
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        super(Critic, self).__init__()

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, 1, activation=nn.Tanh)

    def forward(self, state) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """

        return self.layers(state)


class ActorCritic(PolicyGradient):
    """ Actor-Critic module that handles both actions and values
    Actors and critics here don't share a body but do share a loss
    function. Therefore they are both in the same module
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        action_std: float = 0.0,
    ):
        super(ActorCritic, self).__init__(
            state_dim,
            action_dim,
            hidden_dims,
            action_std
        )

        self.critic = Critic(
            state_dim, action_dim, hidden_dims,
        ).to(get_device())

        print(self)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        pi = self.actor.forward(state)
        mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()
        values = self.critic(state).squeeze(-1)

        return values, action_logprob, entropy, mu, std

    def get_evaluation(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.FloatTensor(state.cpu().numpy()).to(get_device())
        action = torch.FloatTensor(action).to(get_device())

        v, prob, entropy, mu, std = self.evaluate(state, action)

        return (
            v.cpu().data.numpy(),
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth')))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()

class ReplayBuffer(object):
    """ Replay buffer to store transitions. Efficiency could probably be
    improved.

    While it is called a ReplayBuffer, it is not actually one as no "Replay"
    is performed. As it is used by on-policy algorithms, the buffer should
    be cleared every time it is sampled.

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: int, action_dim: int, n_trajectories: int,
        max_traj_length: int, gamma: float, lmbda: float = 0.95
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        n_trajectories: int
            Number of learned accumulating transitions
        max_traj_length: int
            Maximum length of trajectories
        gamma: float
            Discount factor.
        lmbda: float
            GAE factor.
        """
        self.ptr = 0

        self.n_trajectories = n_trajectories
        self.max_traj_length = max_traj_length
        self.device = get_device()
        self.lens = np.zeros((n_trajectories), dtype=np.int32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL Buffers "filled with zeros"
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))
        self.mus = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

    def add(
        self,
        ind: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        probs: np.ndarray,
        mus: np.ndarray,
        stds: np.ndarray,
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        values: np.ndarray
            Batch of "old" value estimates for this batch of transitions
        next_values : np.ndarray
            Batch of "old" value-primes for this batch of transitions
        probs: np.ndarray
            Batch of "old" log-probs for this batch of transitions

        """
        self.state[ind, self.ptr] = state
        self.action[ind, self.ptr] = action

        # These are actually not needed
        self.next_state[ind, self.ptr] = next_state
        self.reward[ind, self.ptr] = reward
        self.not_done[ind, self.ptr] = (1. - done)

        # Values for losses
        self.values[ind, self.ptr] = values
        self.next_values[ind, self.ptr] = next_values
        self.probs[ind, self.ptr] = probs

        self.mus[ind, self.ptr] = mus
        self.stds[ind, self.ptr] = stds

        self.lens[ind] += 1

        for j in range(len(ind)):
            i = ind[j]

            if done[j]:
                # Calculate the expected returns: the value function target
                rew = self.reward[i, :self.ptr]
                # rew = (rew - rew.mean()) / (rew.std() + 1.e-8)
                self.ret[i, :self.ptr] = \
                    self.discount_cumsum(
                        rew, self.gamma)

                # Calculate GAE-Lambda with this trick
                # https://stackoverflow.com/a/47971187
                # TODO: make sure that this is actually correct
                # TODO?: do it the usual way with a backwards loop
                deltas = rew + \
                    (self.gamma * self.next_values[i, :self.ptr] *
                     self.not_done[i, :self.ptr]) - \
                    self.values[i, :self.ptr]

                if self.lmbda == 0:
                    self.adv[i, :self.ptr] = self.ret[i, :self.ptr] - \
                        self.values[i, :self.ptr]
                else:
                    self.adv[i, :self.ptr] = \
                        self.discount_cumsum(deltas, self.gamma * self.lmbda)

        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        # Taken from spinup implementation
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
                vector x,
                [x0,
                 x1,
                 x2]
        output:
                [x0 + discount * x1 + discount^2 * x2,
                 x1 + discount * x2,
                 x2]
        """
        return scipy.signal.lfilter(
            [1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def sample(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Sample all transitions.

        Parameters:
        -----------

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ret: torch.Tensor
            Sampled return estimate, target for V
        adv: torch.Tensor
            Sampled advantges, factor for policy update
        probs: torch.Tensor
            Sampled old action probabilities
        """
        # TODO?: Not sample whole buffer ? Have M <= N*T ?

        # Generate indices
        row, col = zip(*((i, le)
                         for i in range(len(self.lens))
                         for le in range(self.lens[i])))

        s, a, ret, adv, probs, mus, stds = (
            self.state[row, col], self.action[row, col], self.ret[row, col],
            self.adv[row, col], self.probs[row, col], self.mus[row, col],
            self.stds[row, col])

        # Normalize advantage. Needed ?
        # Trick used by OpenAI in their PPO impl
        # adv = (adv - adv.mean()) / (adv.std() + 1.e-8)

        shuf_ind = np.arange(s.shape[0])

        # Shuffling makes the learner unable to track in "two directions".
        # Why ?
        # np.random.shuffle(shuf_ind)

        self.clear_memory()

        return (s[shuf_ind], a[shuf_ind], ret[shuf_ind], adv[shuf_ind],
                probs[shuf_ind], mus[shuf_ind], stds[shuf_ind])

    def clear_memory(self):
        """ Reset the buffer
        """

        self.lens = np.zeros((self.n_trajectories), dtype=np.int32)
        self.ptr = 0

        # RL Buffers "filled with zeros"
        # TODO: Is that actually needed ? Can't just set self.ptr to 0 ?
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))
        self.mus = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

    def __len__(self):
        return np.sum(self.lens)

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

