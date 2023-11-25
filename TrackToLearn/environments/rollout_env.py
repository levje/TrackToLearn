import numpy as np
from typing import Callable
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            stopping_idx: np.ndarray,
            stopping_flags: np.ndarray,
            current_length: int,
            stopping_criteria: dict[StoppingFlags, Callable],
            format_state_func: Callable[[np.ndarray], None],
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
        if True:#backtrackable_mask.size > 0 and backup_length > 1:  # To keep the initial point
            backtrackable_streamlines[:, backup_length:current_length, :] = 0  # Backtrack on backup_length
            rollouts = np.repeat(backtrackable_streamlines[None, ...], self.n_rollouts,
                                 axis=0)  # Contains the different rollouts
            while backup_length < (current_length + self.extra_n_steps):
                for rollout in range(self.n_rollouts):
                    state = format_state_func(rollouts[rollout, :, :backup_length, :])
                    new_directions = self.rollout_agent.select_action(state=state, probabilistic=prob)
                    rollouts[rollout, :, backup_length, :] += new_directions
                backup_length += 1

            # TODO: Get the best rollout for each streamline.
            # TODO: Squash the retained rollouts to the current_length
            # TODO: Replace the original streamlines with the best rollouts.
            # TODO: Change the done state of the streamlines that are valid.

        return streamlines

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
