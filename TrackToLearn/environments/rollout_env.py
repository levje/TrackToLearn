import numpy as np
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.datasets.utils import (MRIDataVolume, SubjectData,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)


class RolloutEnvironment(BaseEnv):
    def __init__(self,
                 input_volume: MRIDataVolume,
                 tracking_mask: MRIDataVolume,
                 target_mask: MRIDataVolume,
                 seeding_mask: MRIDataVolume,
                 peaks: MRIDataVolume,
                 env_dto: dict,
                 agent: ActorCritic,
                 n_rollouts: int = 5,  # Nb of rollouts to try
                 backup_size: int = 1,  # Nb of steps we are backtracking
                 forward_comp: int = 6,  # Nb of steps further we need to compare the different rollouts
                 include_mask: MRIDataVolume = None,
                 exclude_mask: MRIDataVolume = None,
                 ):
        super().__init__(input_volume, tracking_mask, target_mask, seeding_mask, peaks, env_dto, include_mask,
                         exclude_mask)
        self.rollout_agent = agent
        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.forward_comp = forward_comp

    def rollout(
            self,
            streamlines: np.ndarray,
            stopping_idx: np.ndarray,
            stopping_flags: np.ndarray,
            current_length: int,
            prob: float = 0.1
    ):
        backtrackable_mask = self._get_backtrackable_indices(streamlines[stopping_idx, :, :], stopping_flags)
        backtrackable_idx = np.where(backtrackable_mask)[0]
        backtrackable_streamlines = streamlines[backtrackable_idx, :, :]

        # Backtrack the streamlines
        backup_length = current_length - self.backup_size
        if backtrackable_mask.size > 0 and backup_length > 1:  # To keep the initial point
            backtrackable_streamlines[:, backup_length:current_length, :] = 0  # Backtrack on backup_length
            rollouts = np.repeat(backtrackable_streamlines[None, ...], self.n_rollouts,
                                 axis=0)  # Contains the different rollouts
            while backup_length < (current_length + self.forward_comp):
                for rollout in range(self.n_rollouts):
                    state = self._format_state(rollouts[rollout, :, :backup_length, :])
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
