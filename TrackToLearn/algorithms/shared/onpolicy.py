import numpy as np
import torch

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.nn.functional as F

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)

class HybridMaxEntropyActor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list,
        device: torch.device,
        action_std: float = 0.0,
        output_activation=nn.Tanh
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

        """
        NB: This is a modified Actor. We want to be able to use PPO with the same actor as SAC.
        Classically, in PPO, the STD is state-independent, but in SAC, it is state-dependent.
        We modified the PPO's actor to have a state-dependent STD, as in SAC.
        """
        super(HybridMaxEntropyActor, self).__init__()

        self.action_dim = action_dim
        self.hidden_layers = hidden_layers

        self.output_activation = output_activation()

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim * 2)


    def forward(
        self,
        state: torch.Tensor,
        probabilistic: float,
    ) -> torch.Tensor:
        """ Forward propagation of the actor. Log probability is computed
        from the Gaussian distribution of the action and correction
        for the Tanh squashing is applied.

        Parameters:
        -----------
        state: torch.Tensor
            Current state of the environment
        probabilistic: float
            Factor to multiply the standard deviation by when sampling.
            0 means a deterministic policy, 1 means a fully stochastic.
        """

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

        # Compute mean and log_std from neural network. Instead of
        # have two separate outputs, we have one output of size
        # action_dim * 2. The first action_dim are the means, and
        # the last action_dim are the log_stds.
        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]
        # Constrain log_std inside [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # Compute std from log_std
        std = torch.exp(log_std) * probabilistic
        # Sample from Gaussian distribution using reparametrization trick
        pi_distribution = Normal(mu, std, validate_args=False)
        pi_action = pi_distribution.rsample()

        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # Squash correction
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=1)

        # Run actions through tanh to get -1, 1 range
        pi_action = self.output_activation(pi_action)
        # Return action and logprob
        return pi_action, logp_pi, pi_distribution.entropy()

class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list,
        device: torch.device,
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

        self.layers = make_fc_network(
            hidden_layers, state_dim, action_dim, activation=nn.Tanh)

        # State-independent STD, as opposed to SAC which uses a
        # state-dependent STD.
        # See https://spinningup.openai.com/en/latest/algorithms/sac.html
        # in the "You Should Know" box
        log_std = -action_std * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, state: torch.Tensor):
        mu = self.layers(state)
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
        device: torch.device,
        action_std: float = 0.0,
        actor_cls: nn.Module = Actor,
    ):
        super(PolicyGradient, self).__init__()
        self.device = device
        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.actor = actor_cls(
            state_dim, action_dim, self.hidden_layers, action_std,
        ).to(device)

    def act(
        self, state: torch.Tensor, probabilistic: float = 1.0,
    ) -> torch.Tensor:
        """ Select noisy action according to actor
        """
        action, logprob, entropy = self.actor.forward(state, probabilistic)
        return action, logprob, entropy

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, probabilistic: float
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
        self, state: np.array, probabilistic: float = 1.0,
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

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action, _, _ = self.act(state, probabilistic)

        return action

    def get_evaluation(
        self, state: np.array, action: np.array, probabilistic: float = 1.0
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

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

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
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))

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
        hidden_layers: list,
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

        self.layers = make_fc_network(
            hidden_layers, state_dim, 1, activation=nn.Tanh)

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
        device: torch.device,
        action_std: float = 0.0,
        actor_cls: nn.Module = Actor
    ):
        super(ActorCritic, self).__init__(
            state_dim,
            action_dim,
            hidden_dims,
            device,
            action_std,
            actor_cls
        )

        self.critic = Critic(
            state_dim, action_dim, self.hidden_layers,
        ).to(self.device)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, probabilistic: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        _, logp_pi, entropy = self.actor.forward(state, probabilistic)
        values = self.critic(state).squeeze(-1)

        return values, logp_pi, entropy

    def get_evaluation(
        self, state: np.array, action: np.array, probabilistic: float
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

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

        v, prob, entropy = self.evaluate(state, action, probabilistic)

        return (
            v.cpu().data.numpy(),
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy())

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
            torch.load(pjoin(path, filename + '_critic.pth'),
                       map_location=self.device))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))

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

class PPOActorCritic(ActorCritic):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: str, device: torch.device, action_std: float = 0):
        super().__init__(state_dim, action_dim, hidden_dims, device, action_std, actor_cls=HybridMaxEntropyActor)

    def load_policy(self, path: str, filename: str):
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))
