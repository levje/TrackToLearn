import numpy as np
from typing import Callable

import torch

from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.datasets.utils import (MRIDataVolume, SubjectData,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)

from TrackToLearn.environments.reward import Reward
from TrackToLearn.experiment.oracle_validator import OracleValidator
from dipy.tracking.streamline import length, transform_streamlines
import time

from fury import actor, window


class RolloutEnvironment(object):

    # Specific constructor to avoid creating another RolloutEnvironment from the BaseEnv's constructor.
    def __init__(self,
                 agent: ActorCritic = None,
                 n_rollouts: int = 5,  # Nb of rollouts to try
                 backup_size: int = 1,  # Nb of steps we are backtracking
                 extra_n_steps: int = 6,  # Nb of steps further we need to compare the different rollouts
                 max_streamline_steps: int = 256  # Max length of a streamline
                 ):

        self.rollout_agent = agent
        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.extra_n_steps = extra_n_steps
        self.max_streamline_steps = max_streamline_steps

    # Overrides the set_rollout_agent from the BaseEnv
    def set_rollout_agent(self, rollout_agent: ActorCritic):
        self.rollout_agent = rollout_agent

    def is_rollout_agent_set(self):
        return self.rollout_agent is not None

    # TODO: Copied from env
    def _compute_stopping_flags(
        self,
        streamlines: np.ndarray,
        stopping_criteria: dict[StoppingFlags, Callable]
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should
        continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_stop : `numpy.ndarray`
            Boolean array, True is tracking should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=int)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            flags[stopped_by_criterion] |= flag.value
            should_stop[stopped_by_criterion] = True

        return should_stop, flags

    def rollout(
            self,
            streamlines: np.ndarray,
            in_stopping_idx: np.ndarray,
            in_stopping_flags: np.ndarray,
            current_length: int,
            stopping_criteria: dict[StoppingFlags, Callable],
            format_state_func: Callable[[np.ndarray], np.ndarray],
            format_action_func: Callable[[np.ndarray], np.ndarray],
            oracle_reward: Reward,  # TODO: It's not technically a reward. It's just a score to evaluate the best streamline.
            prob: float = 1.1,
            dynamic_prob: bool = False,
            render: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _prob = 2.0  # Copy for dynamic probability adjustment during rollout

        # The agent initialisation should be made in the __init__. The environment should be remodeled to allow
        # creating the agent before the environment.
        if self.rollout_agent is None:
            print("Rollout: no agent specified. Rollout aborted.")
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        # Backtrack the streamlines
        backup_length = current_length - self.backup_size
        initial_backtracked_length = backup_length

        if backup_length == current_length:  # We don't backtrack in that case, there's no need to compute the rest
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        backtrackable_mask = self._get_backtrackable_indices(in_stopping_flags)
        backtrackable_idx = in_stopping_idx[backtrackable_mask]

        if backtrackable_idx.size <= 0 or backup_length <= 1:
            # Can't backtrack, because we're at the start or every streamline ends correctly (in the target).
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags


        streamlines[backtrackable_idx, backup_length:current_length, :] = 0  # Backtrack on backup_length
        rollouts = np.repeat(streamlines[None, ...], self.n_rollouts,
                             axis=0)  # Contains the different rollouts

        max_rollout_length = min(current_length + self.extra_n_steps, self.max_streamline_steps)

        rollouts_continue_idx = [backtrackable_idx for _ in range(self.n_rollouts)]  # np.repeat(backtrackable_idx[None, ...], self.n_rollouts, axis=0)

        # Every streamline is continuing, thus no stopping flag for each of them
        flags = np.zeros((self.n_rollouts, backtrackable_idx.shape[0]), dtype=int)
        true_lengths = np.zeros((self.n_rollouts, backtrackable_idx.shape[0]), dtype=int)

        while backup_length < max_rollout_length and not all(np.size(arr) == 0 for arr in rollouts_continue_idx):

            if dynamic_prob and backup_length > current_length:
                _prob = 0.0  # Once we reached where we previously failed, stop exploring and just follow the agent.

            for rollout in range(self.n_rollouts):

                if rollouts_continue_idx[rollout].size <= 0:
                    # No more streamlines to continue
                    continue

                # Grow continuing streamlines one step forward
                state = format_state_func(rollouts[rollout, rollouts_continue_idx[rollout], :backup_length, :])

                with torch.no_grad():
                    actions = self.rollout_agent.select_action(state=state, probabilistic=_prob).to(device='cpu', copy=True).numpy()
                new_directions = format_action_func(actions)

                # Step forward
                rollouts[rollout, rollouts_continue_idx[rollout], backup_length, :] = rollouts[rollout, rollouts_continue_idx[rollout], backup_length - 1, :] + new_directions

                # Get continuing streamlines that should stop and their stopping flag
                should_stop, new_flags = self._compute_stopping_flags(
                    rollouts[rollout, rollouts_continue_idx[rollout], :backup_length + 1, :],
                    stopping_criteria
                )

                # See which trajectories is stopping or continuing
                new_continue_idx, stopping_idx = (rollouts_continue_idx[rollout][~should_stop],
                                                  rollouts_continue_idx[rollout][should_stop])

                relative_stopping_indices = np.where(np.isin(backtrackable_idx, stopping_idx))[0]
                true_lengths[rollout, relative_stopping_indices] = backup_length

                rollouts_continue_idx[rollout] = new_continue_idx

                # Keep the reason why tracking stopped (don't delete a streamline that reached the target!)

                flags[rollout, relative_stopping_indices] = new_flags[should_stop]

            backup_length += 1

        if current_length > 100:
            self._render_screenshot(initial_backtracked_length, backup_length, streamlines, rollouts, flags, backtrackable_idx, oracle_reward, true_lengths)

        # Get the best rollout for each streamline.
        best_rollouts, new_flags = self._filter_best_rollouts(rollouts, flags, backtrackable_idx, oracle_reward)

        # Squash the retained rollouts to the current_length
        best_rollouts[:, current_length + 1:, :] = 0

        # Replace the original streamlines with the best rollouts.
        streamlines[backtrackable_idx, :, :] = best_rollouts
        in_stopping_flags[backtrackable_mask] = new_flags  # Remove? Should we overwrite the rollouts/streamlines that are still failing?

        mask = in_stopping_flags > 0

        # Get new continuing streamlines' indexes in the full streamlines array
        continuing_rollouts = np.where(new_flags == 0)[0]
        new_continuing_streamlines = in_stopping_idx[continuing_rollouts]

        # Remove stopping flags for the successful rollouts
        in_stopping_idx = in_stopping_idx[mask]

        return streamlines, new_continuing_streamlines, in_stopping_idx, in_stopping_flags

    def _trim_zeros(self, streamlines: np.ndarray, true_lengths: np.ndarray) -> list[np.ndarray]:
        return [streamline[:l] for streamline, l in zip(streamlines, true_lengths)]

    def _render_screenshot(self, backtracked_length: int, rollout_length: int, streamlines: np.ndarray, rollouts: np.ndarray, flags: np.ndarray, backtrackable_idx: np.ndarray, oracle_reward, true_lengths):
        backtracking_streamlines = streamlines[backtrackable_idx, :backtracked_length, :]
        backtracking_rollouts = rollouts[:, backtrackable_idx, :rollout_length, :]
        chosen_streamline = self._trim_zeros(backtracking_rollouts[:, 0, ...], true_lengths[:, 0])


        reference_streamline = backtracking_streamlines[None, 0, ...]
        reference_actor = actor.line(reference_streamline, (1.0, 0, 0))
        rollouts_actor = actor.line(chosen_streamline, (1.0, 0.5, 0))

        scene = window.Scene()
        scene.add(rollouts_actor)
        scene.add(reference_actor)
        window.show(scene, size=(600, 600), reset_camera=False)

        # Select a rollout at random


        pass

    def _filter_best_rollouts(self,
                              rollouts: np.ndarray,
                              flags,
                              backtrackable_idx: np.ndarray,
                              oracle_reward: Reward  # TODO: It's not technically a reward. It's just a score to evaluate the best streamline.
                              ):

        dones = flags > 0
        rollouts_scores = np.zeros((self.n_rollouts, backtrackable_idx.shape[0]), dtype=np.float32)
        for rollout in range(rollouts.shape[0]):
            rewards = oracle_reward(rollouts[rollout, backtrackable_idx, ...], dones=dones[rollout, :])[0]
            rollouts_scores[rollout] = rewards

        # Filter based on the calculated scores and keep the best rollouts and their according flags
        best_rollout_indices = np.argmax(rollouts_scores, axis=0)
        streamline_indices = np.arange(backtrackable_idx.shape[0])
        best_rollouts = rollouts[best_rollout_indices, streamline_indices, ...]
        best_flags = flags[best_rollout_indices, streamline_indices]

        return best_rollouts, best_flags

    @staticmethod
    def _get_backtrackable_indices(
            stopping_flags: np.ndarray
    ) -> np.ndarray:
        """ Filter out the stopping flags from which we are able to perform backtracking to retry different paths
        for each stopped streamline.

        For example, if a streamline stopped because of STOPPING_TARGET, we might not want to backtrack since the
        streamline ends in the gray matter.
        """
        flag1 = np.not_equal(stopping_flags, StoppingFlags.STOPPING_TARGET.value)
        return flag1
