from typing import Optional, Sequence

import numpy as np
import math
import sys

from .base_sampler import Sampler, SampleRange


class NaiveSampler(Sampler):
    def __init__(self, seed=0):
        self.random = np.random.RandomState(seed)

    def reset(self):
        pass

    def sample(self, max_idx: int, s: int):
        """
        Random sample drawn from [0,max_idx]
        :param max_idx: integer defining right bound of range
        :param s: number of samples
        :return: sequence of integer samples
        """
        if s > max_idx:
            return self.random.choice(max_idx, size=max_idx, replace=False)
        else:
            return self.random.choice(max_idx, size=s, replace=False)
