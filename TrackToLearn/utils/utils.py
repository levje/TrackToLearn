from typing import Iterable
import math
import os
import sys

from dipy.core.geometry import sphere2cart
from os.path import join as pjoin
from time import time
import cProfile
import pstats
import io
import numpy as np
import torch

COLOR_CODES = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m'
}


class LossHistory(object):
    """ History of the loss during training.
    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self, name, filename, path):
        self.name = name
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0

        self.filename = filename
        self.path = path

    def __len__(self):
        return len(self.history)

    def update(self, value):
        if np.isinf(value):
            return

        self.history.append(value)
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self, epoch):
        self.epochs.append((epoch, self._avg))
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_epochs += 1

        directory = pjoin(self.path, 'plots')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(pjoin(directory, '{}.npy'.format(self.filename)), 'wb') as f:
            np.save(f, self.epochs)


class Timer:
    """ Times code within a `with` statement, optionally adding color. """

    def __init__(self, txt, newline=False, color=None):
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.10f} sec.".format(time() - self.start))


def from_sphere(actions, sphere, norm=1.):
    vertices = sphere.vertices[actions]
    return vertices * norm


def normalize_vectors(v, norm=1.):
    # v = (v / np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))) * norm
    v = (v / np.sqrt(np.einsum('...i,...i', v, v))[..., None]) * norm
    # assert np.all(np.isnan(v) == False), (v, np.argwhere(np.isnan(v)))
    return v


def from_polar(actions, radius=1.):

    radii = np.ones((actions.shape[0])) * radius
    theta = ((actions[..., 0] + 1) / 2.) * (math.pi)
    phi = ((actions[..., 1] + 1) / 2.) * (2 * math.pi)

    X, Y, Z = sphere2cart(radii, theta, phi)
    cart_directions = np.stack((X, Y, Z), axis=-1)
    return cart_directions


def prettier_metrics(metrics, as_line: bool = False, title: str = None):
    """ Pretty print metrics """
    if as_line:
        return " | ".join(["{}: {:.4f}".format(k, v) for k, v in metrics.items()])

    # Build a string of the metrics to eventually print
    # The string should be the representation of a table
    # with each metrics on a row with the following format:
    # ===================================
    # Test results
    # ===================================
    # | metric_name     |   metric_value |
    # | metric_name     |   metric_value |
    # ===================================

    # Get the length of the longest metric value and name
    max_key_len = max([len(k) for k in metrics.keys()])
    max_val_len = max([len(str(round(v, 4))) for v in metrics.values()])

    # Create the header
    header = "=" * (max_key_len + max_val_len + 7)
    if title is None:
        header = header + "\nTest results\n"
    else:
        header = header + "\n" + title + "\n"
    header = header + "=" * (max_key_len + max_val_len + 7)

    # Create the table
    table = ""
    for k, v in sorted(metrics.items()):
        table = table + \
            "\n| {:{}} | {:.4f} |".format(
                k, max_key_len, round(v, 4), max_val_len)

    # Create the footer
    footer = "=" * (max_key_len + max_val_len + 7)

    return header + table + "\n" + footer

def prettier_dict(d: dict, title: str = None):
    # Build a string of the metrics to eventually print
    # The string should be the representation of a table
    # with each metrics on a row with the following format:
    # ===================================
    # <Print the title here>
    # ===================================
    # | train
    # |   ↳ other_dict
    # |     ↳ value1 : 0.1
    # |     ↳ value2 : 0.2
    # |   ↳ value3 : 0.3
    # | test
    # |   ↳ other_dict
    # |     ↳ value1 : 0.1
    # |     ↳ value2 : 0.2
    # |   ↳ value4 : 0.4
    # ===================================

    # We want to recursively print the dictionary in a pretty way
    # with increasing indentation for each level of the dictionary.
    # At level 0, there should only be | and one whitespace before the key.
    # At level 1, there should be |, 2 whitespaces, ↳ and one whitespace before the key.
    # At level 2, there should be |, 4 whitespaces, ↳ and one whitespace before the key.
    # etc.
    left_padding = 1        # Padding after |
    nb_indent_spaces = 2    # Whitespace after left padding and before ↳
    right_padding = 4       # Number of extra "=" characters for the header/footer

    def pretty(d, level=0):
        table = ""
        indentation = " " * nb_indent_spaces * level
        for k, v in d.items():
            if isinstance(v, dict):
                if level > 0:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + "↳ " + k + pretty(v, level + 1)
                else:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + k + pretty(v, level + 1)
            else:
                if level > 0:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + "↳ " + k + " : " + str(v)
                else:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + k + " : " + str(v)
        return table
    
    table = pretty(d)

    # Create the header. The header length should be as long as the longest line in the table.
    max_line_len = max([len(line) for line in table.split("\n")])
    if title is not None:
        header = "=" * (max_line_len + right_padding) + "\n"
        header = header + title + "\n"
        header = header + "=" * (max_line_len + right_padding)
    else:
        header = "=" * (max_line_len + right_padding)

    # Create the footer
    footer = "=" * (max_line_len + right_padding)

    return header + table + "\n" + footer


class TTLProfiler:
    def __init__(self, enabled: bool = True, throw_at_stop: bool = True) -> None:
        self.pr = None
        self.enabled = enabled
        self.throw_at_stop = throw_at_stop

    def start(self):
        if not self.enabled:
            return

        if self.pr is not None:
            import warnings
            warnings.warn(
                "Profiler already started. Stop it before starting a new one.")
            return

        self.pr = cProfile.Profile()
        self.pr.enable()

    def stop(self):
        if not self.enabled:
            return

        if self.pr is None:
            import warnings
            warnings.warn("Profiler not started, but stop() was called.")
            return

        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
        self.pr = None

        if self.throw_at_stop:
            raise RuntimeError("Profiling stopped.")


def break_if_grads_have_nans(named_params: Iterable):
    for name, param in named_params:
        if param.grad is not None and torch.isnan(param.grad).any():
            breakpoint()
            indexes = get_index_where_nans(param.grad)
            print(f"Gradient of parameter {name} has NaNs.")
            raise ValueError('Gradient has NaNs')


def break_if_params_have_nans(params: Iterable):
    for p in params:
        if torch.isnan(p).any():
            breakpoint()
            indexes = get_index_where_nans(p)
            print("Parameter has NaNs.")
            raise ValueError('Parameter has NaNs')
        elif torch.isinf(p).any():
            breakpoint()
            indexes = torch.isinf(p).nonzero()
            print("Parameter has Infs.")
            raise ValueError('Parameter has Infs')


def break_if_found_nans(t: torch.Tensor):
    if isinstance(t, torch.Tensor) and torch.numel(t) != 0:
        # Check if there's a NaN
        if torch.isnan(t).any():
            breakpoint()
            indexes = get_index_where_nans(t)
            print("Tensor has NaNs.")
            raise ValueError('Tensor has NaNs')
        # Check if there's any infinity
        elif torch.isinf(t).any():
            breakpoint()
            indexes = torch.isinf(t).nonzero()
            print("Tensor has Infs.")
            raise ValueError('Tensor has Infs')


def break_if_found_nans_args(*args):
    for arg in args:
        break_if_found_nans(arg)


def get_index_where_nans(t: torch.Tensor):
    if torch.numel(t) != 0:
        if torch.isnan(t.max()) or torch.isnan(t.min()):
            return torch.isnan(t).nonzero()
    return torch.tensor([], dtype=torch.int32)
