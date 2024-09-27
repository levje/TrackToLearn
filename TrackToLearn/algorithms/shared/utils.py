import numpy as np
import torch

from torch import nn
from collections import defaultdict


def add_item_to_means(means, dic):
    if isinstance(means, defaultdict):
        for k in dic.keys():
            means[k].append(dic[k])
    else:
        means = {k: means[k] + [dic[k]] for k in dic.keys()}
    return means


def add_to_means(means, dic):
    return {k: means[k] + dic[k] for k in dic.keys()}

# TODO: Remove that, it's just to test classic ppo implementation
def old_mean_losses(dic):
    return {k: np.mean(dic[k]) for k in dic.keys()}

def get_mean_item(dic, key):
    return np.mean(torch.stack(dic[key]).cpu().numpy())

def mean_losses(dic):
    new_dict = {k: np.mean(torch.stack(dic[k]).cpu().numpy(), axis=0)
                for k in dic.keys()}
    return new_dict

def add_losses(dic):
    new_dict = {k: np.sum(torch.stack(dic[k]).cpu().numpy(), axis=0)
                for k in dic.keys()}
    return new_dict

def mean_rewards(dic):
    return {k: np.mean(np.asarray(dic[k]), axis=0) for k in dic.keys()}


def harvest_states(i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)


def format_widths(widths_str):
    return np.asarray([int(i) for i in widths_str.split('-')])


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation()])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size)])
    return nn.Sequential(*layers)
