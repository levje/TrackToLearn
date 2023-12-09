import numpy as np
from typing import Callable

import torch
import nibabel as nib
from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.datasets.utils import (MRIDataVolume, SubjectData,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)

from TrackToLearn.environments.reward import Reward, RewardFunction
from TrackToLearn.experiment.oracle_validator import OracleValidator
from dipy.tracking.streamline import length, transform_streamlines
import time

from fury import actor, window


class RolloutEnvironment(object):

    def __init__(self,
                 reference: str,
                 agent: ActorCritic = None,
                 n_rollouts: int = 5,  # Nb of rollouts to try
                 backup_size: int = 1,  # Nb of steps we are backtracking
                 extra_n_steps: int = 6,  # Nb of steps further we need to compare the different rollouts
                 max_streamline_steps: int = 256,  # Max length of a streamline
                 oracle_reward: Reward = None,
                 reward_function: RewardFunction = None
                 ):

        self.rollout_agent = agent
        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.extra_n_steps = extra_n_steps
        self.max_streamline_steps = max_streamline_steps
        self.reference = nib.load(reference)
        self.reference_affine = self.reference.affine
        self.reference_data = self.reference.get_fdata()
        self.oracle_reward = oracle_reward
        self.reward_function = reward_function

    # Specific constructor to avoid creating another RolloutEnvironment from the BaseEnv's constructor.

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
            prob: float = 1.1,
            dynamic_prob: bool = False,
            render: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _prob = 1.1  # Copy for dynamic probability adjustment during rollout

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

        # TODO: REMOVEEEEEEEEEEEEE
        # backtrackable_mask[0] = False

        backtrackable_idx = in_stopping_idx[backtrackable_mask]

        if backtrackable_idx.size <= 0 or backup_length <= 1:
            # Can't backtrack, because we're at the start or every streamline ends correctly (in the target).
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        backtracked_streamlines = streamlines.copy()
        backtracked_streamlines[backtrackable_idx, backup_length:current_length, :] = 0  # Backtrack on backup_length
        rollouts = np.repeat(backtracked_streamlines[None, ...], self.n_rollouts,
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
                true_lengths[rollout, relative_stopping_indices] = backup_length + 1

                rollouts_continue_idx[rollout] = new_continue_idx

                # Keep the reason why tracking stopped (don't delete a streamline that reached the target!)

                flags[rollout, relative_stopping_indices] = new_flags[should_stop]

            backup_length += 1

        true_lengths[true_lengths == 0] = backup_length

        # Get the best rollout for each streamline.
        # TODO: Since sometimes the best rollout is shorter than the current length, we will squash by keeping the smallest, thus introducing some zeros along the way.
        best_rollouts, new_flags, best_true_lengths, rollouts_scores = self._filter_best_rollouts(rollouts, flags, backtrackable_idx, true_lengths)
        streamline_improvement_idx = self._filter_worse_rollouts(current_length, best_rollouts, best_true_lengths, flags)

        # if current_length > 5 and current_length < 150:
           # self._render_screenshot(initial_backtracked_length, backup_length, current_length, streamlines, rollouts, flags, backtrackable_idx, true_lengths, rollouts_scores)

        # Squash the retained rollouts to the current_length
        best_rollouts[:, current_length:, :] = 0

        # Replace the original streamlines with the best rollouts.
        streamlines[backtrackable_idx[streamline_improvement_idx], :, :] = best_rollouts[streamline_improvement_idx]
        # TODO: USE THE streamline_improvement_idx to only change the flags of the streamlines that improved
        backtrackable_relative_to_in_stopping_idx = np.where(backtrackable_mask)[0]
        in_stopping_flags[backtrackable_relative_to_in_stopping_idx[streamline_improvement_idx]] = new_flags[streamline_improvement_idx]  # Remove? Should we overwrite the rollouts/streamlines that are still failing?

        mask_improvement = np.full(in_stopping_idx.shape, False)
        mask_improvement[backtrackable_relative_to_in_stopping_idx[streamline_improvement_idx]] = True

        # Get new continuing streamlines' indexes in the full streamlines array
        continuing_rollouts = np.where(np.logical_and(new_flags == 0, mask_improvement[backtrackable_relative_to_in_stopping_idx]))[0]
        new_continuing_streamlines = in_stopping_idx[continuing_rollouts]

        # Remove stopping flags for the successful rollouts
        in_stopping_idx = in_stopping_idx[np.where(in_stopping_flags > 0)[0]]

        assert in_stopping_idx.shape[0] == np.sum(in_stopping_flags > 0)

        self._check_if_streamline_has_coordinate_zero(streamlines)
        self._check_if_continuing_streamline_is_zero(streamlines, current_length, new_continuing_streamlines)

        return streamlines, new_continuing_streamlines, in_stopping_idx, in_stopping_flags

    def _check_if_continuing_streamline_is_zero(self, streamlines, current_length, continuing_streamlines):

        current_pos_is_null = np.all(streamlines[continuing_streamlines, current_length] == [0.0, 0.0, 0.0], axis=1)
        num_current_pos_is_null = current_pos_is_null.sum()
        assert num_current_pos_is_null == continuing_streamlines.shape[0]

        previous_pos_is_null = np.all(streamlines[continuing_streamlines, current_length-1] == [0.0, 0.0, 0.0], axis=1)
        num_prev_pos_is_null = previous_pos_is_null.sum()
        assert num_prev_pos_is_null == 0

    def _check_if_streamline_has_coordinate_zero(self, streamlines):
        zero_mask = np.all(streamlines == [0.0, 0.0, 0.0], axis=2)

        # Shift the zero_mask to the right by one, padding with False
        shifted_zero_mask = np.hstack([np.full((zero_mask.shape[0], 1), False), zero_mask[:, 1:]])

        # Create a mask for non-zero coordinates
        non_zero_mask = ~np.all(streamlines == [0.0, 0.0, 0.0], axis=2)

        # Find where a zero is followed by a non-zero
        zero_followed_by_non_zero = shifted_zero_mask & non_zero_mask

        # Identify streamlines that contain at least one such pattern
        selected_streamlines = np.any(zero_followed_by_non_zero, axis=1)

        # Indices of selected streamlines
        selected_indices = np.where(selected_streamlines)[0]

        assert selected_indices.shape[0] == 0, "Streamlines with zero coordinates found at indices: {}".format(selected_indices)


    def _trim_zeros(self, streamlines: np.ndarray, true_lengths: np.ndarray) -> list[np.ndarray]:
        return [streamline[:l] for streamline, l in zip(streamlines, true_lengths)]

    def _render_screenshot(self, backtracked_length: int, rollout_length: int, current_length, streamlines: np.ndarray, rollouts: np.ndarray, flags: np.ndarray, backtrackable_idx: np.ndarray, true_lengths, rollout_scores: np.ndarray):
        original_streamline = streamlines[backtrackable_idx, :current_length, :]

        backtracking_streamlines = streamlines[backtrackable_idx, :backtracked_length, :]
        backtracking_rollouts = rollouts[:, backtrackable_idx, :rollout_length, :]
        chosen_streamline = self._trim_zeros(backtracking_rollouts[:, 0, ...], true_lengths[:, 0])
        chosen_streamline_scores = rollout_scores[:, 0]
        max_score_idx = np.argmax(chosen_streamline_scores)
        best_streamline = [chosen_streamline[max_score_idx]]
        chosen_streamline.pop(max_score_idx)

        fa_actor = actor.slicer(self.reference_data, opacity=0.7, interpolation='nearest')

        original_reference_streamline = original_streamline[None, 0, ...]
        reference_streamline = backtracking_streamlines[None, 0, ...]
        original_actor = actor.line(original_reference_streamline, (65/255, 143/255, 205/255), linewidth=3)
        reference_actor = actor.line(reference_streamline, (65/255, 143/255, 205/255), linewidth=3)
        rollouts_actor = actor.line(chosen_streamline, (1, 0, 0), linewidth=3)
        best_rollout_actor = actor.line(best_streamline, (0, 1, 0), linewidth=3)

        scene = window.Scene()
        scene.add(fa_actor)
        scene.add(rollouts_actor)
        scene.add(best_rollout_actor)
        scene.add(reference_actor)
        scene.add(original_actor)

        record = True
        window.show(scene, size=(600, 600), reset_camera=False)
        # if record:
            # window.record(scene, out_path='/home/local/USHERBROOKE/levj1404/Desktop/renders/render_rollout_{}_{}.png'.format(backtracked_length, rollout_length), size=(600, 600))
    def _filter_best_rollouts(self,
                              rollouts: np.ndarray,
                              flags,
                              backtrackable_idx: np.ndarray,
                              true_lengths: np.ndarray
                              ):

        dones = flags > 0
        rollouts_scores = np.zeros((self.n_rollouts, backtrackable_idx.shape[0]), dtype=np.float32)
        for rollout in range(rollouts.shape[0]):
            rewards = self.reward_function(rollouts[rollout, backtrackable_idx, ...], dones=dones[rollout, :])[0]
            rollouts_scores[rollout] = rewards

        # Filter based on the calculated scores and keep the best rollouts and their according flags
        best_rollout_indices = np.argmax(rollouts_scores, axis=0)
        streamline_indices = np.arange(backtrackable_idx.shape[0])
        best_rollouts = rollouts[best_rollout_indices, backtrackable_idx, ...]
        best_flags = flags[best_rollout_indices, streamline_indices]
        best_true_lengths = true_lengths[best_rollout_indices, streamline_indices]

        return best_rollouts, best_flags, best_true_lengths, rollouts_scores  # Remove the return of the scores

    def _filter_worse_rollouts(self, current_length: int, best_rollouts: np.ndarray, true_lengths: np.ndarray, flags: np.ndarray):
        # TODO: If the best streamline is smaller, we shouldn't keep it
        # TODO: Unless it has a stopping flag of STOPPING_TARGET
        # best_rollouts is of shape (n_backtrackable, max_length, 3)

        rollouts_long_enough_idx = np.where(true_lengths >= current_length)[0]
        # best_rollouts = best_rollouts[rollouts_long_enough_idx]
        # true_lengths = true_lengths[rollouts_long_enough_idx]
        # flags = flags[rollouts_long_enough_idx]
        return rollouts_long_enough_idx

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
