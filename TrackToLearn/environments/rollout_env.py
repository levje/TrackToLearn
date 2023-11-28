import numpy as np
from typing import Callable

import torch

from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.datasets.utils import (MRIDataVolume, SubjectData,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)


class RolloutEnvironment(object):

    # Specific constructor to avoid creating another RolloutEnvironment from the BaseEnv's constructor.
    def __init__(self,
                 agent: ActorCritic = None,
                 n_rollouts: int = 5,  # Nb of rollouts to try
                 backup_size: int = 1,  # Nb of steps we are backtracking
                 extra_n_steps: int = 6,  # Nb of steps further we need to compare the different rollouts
                 ):


        self.rollout_agent = agent
        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.extra_n_steps = extra_n_steps

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

    def get_streamline_score(self, streamline: np.ndarray):
        # TODO: Implement with the oracle

        return 0.0


    def rollout(
            self,
            streamlines: np.ndarray,
            stopping_idx: np.ndarray,
            stopping_flags: np.ndarray,
            current_length: int,
            stopping_criteria: dict[StoppingFlags, Callable],
            format_state_func: Callable[[np.ndarray], np.ndarray],
            format_action_func: Callable[[np.ndarray], np.ndarray],
            prob: float = 0.1
    ):
        # The agent initialisation should be made in the __init__. The environment should be remodeled to allow
        # creating the agent before the environment.
        if self.rollout_agent is None:
            print("Rollout: no agent specified. Rollout aborted.")
            return streamlines

        backtrackable_mask = self._get_backtrackable_indices(stopping_flags)
        backtrackable_idx = np.where(backtrackable_mask)[0]
        backtrackable_streamlines = streamlines[backtrackable_idx, :, :]

        # Backtrack the streamlines
        backup_length = current_length - self.backup_size
        if backtrackable_mask.size > 0: # and backup_length > 1:  # To keep the initial point
            backtrackable_streamlines[:, backup_length:current_length, :] = 0  # Backtrack on backup_length
            rollouts = np.repeat(backtrackable_streamlines[None, ...], self.n_rollouts,
                                 axis=0)  # Contains the different rollouts

            max_rollout_length = current_length + self.extra_n_steps

            rollouts_continue_idx = np.repeat(backtrackable_idx[None, ...], self.n_rollouts, axis=0)

            while backup_length < max_rollout_length:
                for rollout in range(self.n_rollouts):

                    # Grow continuing streamlines one step forward
                    state = format_state_func(rollouts[rollout, rollouts_continue_idx[rollout, ...], :backup_length, :])

                    with torch.no_grad():
                        actions = self.rollout_agent.select_action(state=state, probabilistic=prob).to(device='cpu', copy=True).numpy()
                    new_directions = format_action_func(actions)
                    rollouts[rollout, :, backup_length, :] = np.add(rollouts[rollout, :, backup_length - 1, :], new_directions)

                    # Get stopping and keeping indexes.
                    should_stop, new_flags = self._compute_stopping_flags(
                        rollouts[rollout, rollouts_continue_idx[rollout, ...], :backup_length + 1, :],
                        stopping_criteria
                    )

                    # I'm here
                    # See which trajectories is stopping or continuing
                    self.not_stopping = np.logical_not(should_stop)
                    self.new_continue_idx, self.stopping_idx = (rollouts_continue_idx[rollout, ~should_stop],
                                                                rollouts_continue_idx[rollout, should_stop])

                    # Keep the reason why tracking stopped (don't delete a streamline that reached the target!)
                    self.flags[self.stopping_idx] = new_flags[should_stop]

                backup_length += 1



            # Get the best rollout for each streamline.
            best_rollouts = self._filter_best_rollouts(rollouts)

            # Squash the retained rollouts to the current_length
            best_rollouts[:, current_length:, :] = 0

            # TODO: Replace the original streamlines with the best rollouts.
            valid_rollouts_idx = best_rollouts

            # TODO: Change the done state of the streamlines that are valid.

        return streamlines  # TODO: also return the new stopping and the new flags

    def _filter_best_rollouts(self,
                              all_rollouts: np.ndarray):

        # rollout_scores = np.zeros(self.n_rollouts)
        # rollout_scores[all_rollouts] = self.get_streamline_score(all_rollouts[rollout, :, :, :])

        # This should use the oracle
        return all_rollouts[0, ...]

    @staticmethod
    def _get_backtrackable_indices(
            stopping_flags: np.ndarray
    ) -> np.ndarray:
        """ Filter out the stopping flags from which we are able to perform backtracking to retry different paths
        for each stopped streamline.

        For example, if a streamline stopped because of STOPPING_TARGET, we might not want to backtrack since the
        streamline ends in the gray matter.
        """
        flag1 = np.not_equal(stopping_flags, StoppingFlags.STOPPING_TARGET)
        return flag1
