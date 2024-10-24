import numpy as np
from typing import Callable

import torch
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points
from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)
from TrackToLearn.oracles.transformer_oracle import LightningLikeModule

class RolloutEnvironment(object):

    def __init__(self,
                 reference: nib.Nifti1Image,
                 oracle: LightningLikeModule,
                 n_rollouts: int = 20,  # Nb of rollouts to try
                 backup_size: int = 2,  # Nb of steps we are backtracking
                 extra_n_steps: int = 6,  # Nb of steps further we need to compare the different rollouts
                 min_streamline_steps: int = 1,  # Min length of a streamline
                 max_streamline_steps: int = 256  # Max length of a streamline
                 ):

        self.rollout_agent = None
        self.reference = reference
        self.oracle = oracle

        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.extra_n_steps = extra_n_steps
        self.min_streamline_steps = min_streamline_steps
        self.max_streamline_steps = max_streamline_steps


    def setup_rollout_agent(self, agent: ActorCritic):
        self.rollout_agent = agent

    def _verify_rollout_agent(self):
        if self.rollout_agent is None:
            raise ValueError("Rollout agent not set. Please call setup_rollout_agent before running rollouts.")

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
            prob: float = 1.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._verify_rollout_agent()

        assert self.max_streamline_steps == streamlines.shape[1]

        # Backtrack streamline length
        backup_length = current_length - self.backup_size

        # If the original streamline isn't long enough to backtrack, just do nothing.
        # We don't backtrack in that case, there's no need to compute the rest
        if backup_length < 1 or backup_length == current_length:  
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        # If some streamlines are stopping in the gray matter, we don't want
        # to backtrack them, since they are probably valid. We only want to
        # backtrack prematurely stopped streamlines. (e.g. out of mask).
        backtrackable_mask = self._get_backtrackable_mask(in_stopping_flags)
        backtrackable_idx = in_stopping_idx[backtrackable_mask]

        if backtrackable_idx.size <= 0:
            # Can't backtrack, because we're at the start or every streamline ends correctly (in the target).
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        b_streamlines = streamlines[backtrackable_idx].copy()

        b_streamlines[:, backup_length:current_length, :] = 0  # Backtrack on backup_length
        rollouts = np.repeat(b_streamlines[None, ...], self.n_rollouts,
                             axis=0)  # (n_rollouts, n_streamlines, n_points, 3)

        max_rollout_length = min(current_length + self.extra_n_steps, self.max_streamline_steps)
        
        # List of np.array of size n_rollouts x (n_streamlines,)
        rollouts_continue_idx = [np.arange(b_streamlines.shape[0]) for _ in range(self.n_rollouts)]

        # Every streamline is continuing, thus no stopping flag for each of them
        flags = np.zeros((self.n_rollouts, b_streamlines.shape[0]), dtype=np.int32) # (n_rollouts, n_streamlines)
        true_lengths = np.full((self.n_rollouts, b_streamlines.shape[0]), backup_length - 1, dtype=np.int32) # (n_rollouts, n_streamlines)

        while backup_length < max_rollout_length and not all(np.size(arr) == 0 for arr in rollouts_continue_idx):

            for rollout in range(self.n_rollouts):
                r_continue_idx = rollouts_continue_idx[rollout]

                if r_continue_idx.size <= 0:
                    # No more streamlines to continue for that rollout
                    continue

                # Grow continuing streamlines one step forward
                state = format_state_func(
                    rollouts[rollout, r_continue_idx, :backup_length, :])

                with torch.no_grad():
                    actions = self.rollout_agent.select_action(state, prob)
                    actions = actions.cpu().numpy()
                new_directions = format_action_func(actions)

                # Step forward
                rollouts[rollout, r_continue_idx, backup_length, :] = \
                    rollouts[rollout, r_continue_idx, backup_length - 1, :] + new_directions

                # Get continuing streamlines that should stop and their stopping flag
                should_stop, new_flags = self._compute_stopping_flags(
                    rollouts[rollout, r_continue_idx, :backup_length + 1, :],
                    stopping_criteria
                )

                # See which trajectories is stopping or continuing
                new_continue_idx, stopping_idx = (r_continue_idx[~should_stop],
                                                  r_continue_idx[should_stop])

                true_lengths[rollout, r_continue_idx] = backup_length

                rollouts_continue_idx[rollout] = new_continue_idx

                # Keep the reason why tracking stopped (don't delete a streamline that reached the target!)
                flags[rollout, stopping_idx] = new_flags[should_stop]

            backup_length += 1

        # Get the best rollout for each streamline.
        best_rollouts, new_flags, best_true_lengths = \
            self._filter_best_rollouts(rollouts, flags, backtrackable_idx, true_lengths)
        streamline_improvement_idx = self._get_improvement_idx(current_length, best_true_lengths, flags)

        # Squash the retained rollouts to the current_length
        best_rollouts[:, current_length:, :] = 0

        # Replace the original streamlines with the best rollouts.
        streamlines[backtrackable_idx[streamline_improvement_idx], :, :] = best_rollouts[streamline_improvement_idx]

        # Get the indices of the streamlines that were "saved" and are not
        # stopping anymore, which would be the new continuing streamlines.
        # The new flags should be 0 (not stopped) and it should be an index
        # of a rollout streamline that was improved.
        continuing_rollouts = np.where(
            np.logical_and(
                new_flags == 0, 
                np.isin(backtrackable_idx, backtrackable_idx[streamline_improvement_idx])
            )
        )[0]

        new_continuing_streamlines = in_stopping_idx[continuing_rollouts]

        # Update the stopping flags of the streamlines that were changed.
        # For example, streamlines that aren't stopping anymore have their
        # flags reset to 0, while if we have a streamline that is now stopping
        # in the gray matter (STOPPING_TARGET), that new flag is kept.
        indices = np.arange(in_stopping_idx.shape[0])[backtrackable_mask][streamline_improvement_idx]
        in_stopping_flags[indices] = \
            new_flags[streamline_improvement_idx]

        # Remove the stopping indices that are not stopping anymore
        in_stopping_idx = in_stopping_idx[in_stopping_flags > 0]

        assert in_stopping_idx.shape[0] == np.sum(in_stopping_flags > 0)

        return streamlines, new_continuing_streamlines, in_stopping_idx, in_stopping_flags
    
    @staticmethod
    def _padded_streamlines_to_array_sequence(
            streamlines: np.ndarray,
            true_lengths: np.ndarray) -> ArraySequence:
        """ Convert padded streamlines to an ArraySequence object.

        streamlines: np.ndarray of shape (n_streamlines, max_nb_points, 3)
            containing the streamlines with padding up to max_nb_points.
        true_lengths: np.ndarray of shape (n_streamlines,) containing the
            effective number of points in each streamline.
        """
        assert true_lengths[true_lengths == 0].size == 0, \
            "Streamlines should at least have one point."

        total_points = np.sum(true_lengths)
        _data = np.zeros((total_points, 3), dtype=np.float32)
        _offsets = np.zeros_like(true_lengths)
        _offsets[1:] = np.cumsum(true_lengths)[:-1]
        _lengths = true_lengths

        for i, (streamline, length) in enumerate(zip(streamlines, true_lengths)):
            _data[_offsets[i]:_offsets[i] + length] = streamline[:length]

        array_seq_streamlines = ArraySequence()
        array_seq_streamlines._data = _data
        array_seq_streamlines._offsets = _offsets
        array_seq_streamlines._lengths = _lengths

        return array_seq_streamlines


    def _filter_best_rollouts(self,
                              rollouts: np.ndarray,
                              flags,
                              backtrackable_idx: np.ndarray,
                              true_lengths: np.ndarray
                              ):
        """ Filter the best rollouts based on the oracle's predictions.
        """

        rollouts_scores = np.zeros((self.n_rollouts, rollouts.shape[1]), dtype=np.float32)
        for rollout in range(rollouts.shape[0]):
            # Here, the streamlines should be trimmed to their true length
            # which might differ from one streamline to another.
            # We might want to use nibabel's ArraySequence to store the
            # streamlines so we can also resample them to a fixed number of
            # points using dipy's set_number_of_points function.
            array_seq_streamlines = \
                self._padded_streamlines_to_array_sequence(
                    rollouts[rollout], true_lengths[rollout])
            
            array_seq_streamlines = set_number_of_points(array_seq_streamlines, 128)

            scores = self.oracle.predict(array_seq_streamlines)
            rollouts_scores[rollout] = scores

        # Filter based on the calculated scores and keep the best rollouts and their according flags
        best_rollout_indices = np.argmax(rollouts_scores, axis=0)
        rows = np.arange(rollouts.shape[1]) # Req. for advanced indexing
        best_rollouts = rollouts[best_rollout_indices, rows]
        best_flags = flags[best_rollout_indices, rows]
        best_true_lengths = true_lengths[best_rollout_indices, rows]

        return best_rollouts, best_flags, best_true_lengths

    @staticmethod
    def _get_improvement_idx(current_length: int, rollout_lengths: np.ndarray, flags: np.ndarray):
        # If the best streamline is smaller, we shouldn't keep it
        # TODO: Unless it has a stopping flag of STOPPING_TARGET

        rollouts_long_enough_idx = np.where(rollout_lengths >= current_length)[0]
        return rollouts_long_enough_idx

    @staticmethod
    def _get_backtrackable_mask(
            stopping_flags: np.ndarray
    ) -> np.ndarray:
        """ Filter out the stopping flags from which we are able to perform backtracking to retry different paths
        for each stopped streamline.

        For example, if a streamline stopped because of STOPPING_TARGET, we might not want to backtrack since the
        streamline ends in the gray matter.
        """
        flag1 = np.not_equal(stopping_flags, StoppingFlags.STOPPING_TARGET.value)
        return flag1