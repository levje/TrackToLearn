import math
import os
import sys

from dipy.core.geometry import sphere2cart
from os.path import join as pjoin
from time import time
import cProfile, pstats, io
import numpy as np

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

        print("{:.2f} sec.".format(time() - self.start))


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
            warnings.warn("Profiler already started. Stop it before starting a new one.")
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