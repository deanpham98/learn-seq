from copy import deepcopy

import numpy as np

from gym.spaces import Discrete, Space


class DynamicDiscrete(Space):
    """
    Implement a discrete action spaces that generate sample
    based on a condition on the state.

    It is assumed that there is a finite number of subspaces A_i(s), and that
    n = |union(A_i(s))|

    The user provide n and the list of indices of the actions
    in a subspace A_i(s)

    E.g: A_1(s) = (1, 2, 3, 4), A_2(s) = (3, 5, 6, 7)
    then n = 7, sub_indices = [[1,2,3,4], [3,5,6,7]]
    """
    def __init__(self, n, sub_indices):
        assert n >= 0
        self.n = n
        self.sub_spaces = []
        self.sub_indices = sub_indices
        for s in sub_indices:
            sub_n = len(s)
            self.sub_spaces.append(Discrete(sub_n))
        super(DynamicDiscrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def sample_subspace(self, idx):
        action_idx = self.np_random.randint(self.sub_spaces[idx].n)
        return self.sub_indices[idx][action_idx]

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
                x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "DynamicDiscrete({}, {})".format(len(self.sub_indices[0]),
                                                len(self.sub_indices[1]))

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n


def append_wrapper(config, wrapper, wrapper_kwargs, pos="last"):
    """Append a new wrapper to the current wrapper config. If the current
    wrapper is not a list, convert it to a list"""
    newconf = deepcopy(config)
    old_wrapper = config["wrapper"]
    old_kwargs = config["wrapper_kwargs"]
    new_wrapper = []
    new_wrapper_kwargs = []
    if not isinstance(old_wrapper, list):
        new_wrapper.append(old_wrapper)
        new_wrapper_kwargs.append(old_kwargs)
    else:
        new_wrapper = [wi for wi in old_wrapper]
        new_wrapper_kwargs = [wi for wi in old_kwargs]

    # append new wrapper
    if not isinstance(wrapper, list):
        if pos == "last":
            new_wrapper.append(wrapper)
            new_wrapper_kwargs.append(wrapper_kwargs)
        elif pos == "first":
            new_wrapper.insert(0, wrapper)
            new_wrapper_kwargs.insert(0, wrapper_kwargs)
    else:
        if pos == "last":
            new_wrapper = new_wrapper + wrapper
            new_wrapper_kwargs = new_wrapper_kwargs + wrapper_kwargs
        elif pos == "first":
            new_wrapper = wrapper + new_wrapper
            new_wrapper_kwargs = wrapper_kwargs + new_wrapper_kwargs
    newconf["wrapper"] = new_wrapper
    newconf["wrapper_kwargs"] = new_wrapper_kwargs
    return newconf
