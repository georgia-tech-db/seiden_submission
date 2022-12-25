from typing import Optional, Sequence

import numpy as np
import math
import sys

from .base_sampler import Sampler, SampleRange


class ImportanceSampler(Sampler):
    def __init__(self, seed=0, mixing_eps=0.10):
        self.mixing_eps = mixing_eps
        self.seed = 0
        self.random = np.random.RandomState(seed)
        self.raw_weights = None
        self.weights = None

        self.sampled_idxs = None
        self.sampled_weights = None

    def reset(self):
        self.sampled_idxs = None
        self.sampled_weights = None

    def set_weights(self, weights: np.ndarray):
        self.raw_weights = weights

        scaled_probs = weights / np.sum(weights)
        uniform_prob = 1/len(weights)
        mixed_probs = scaled_probs * (1-self.mixing_eps) + uniform_prob * self.mixing_eps
        self.weights = mixed_probs

    def sample(self, max_idx: int, s: int):
        """
        Random sample drawn from [0,max_idx] importance sampled according to weights
        :param max_idx: integer defining right bound of range
        :param s: number of samples
        :return: sequence of integer samples
        """
        if s > max_idx:
            s = max_idx
        if max_idx > len(self.weights):
            max_idx = len(self.weights)
            weights = self.weights
        elif max_idx < len(self.weights):
            weights = self.weights[:max_idx]
            weights /= weights.sum()
        else:
            weights = self.weights
        self.sampled_idxs = self.random.choice(max_idx, size=s, replace=True, p=weights)
        self.sampled_weights = self.weights[self.sampled_idxs]
        return self.sampled_idxs

    def get_sample_weights(self):
        return self.sampled_weights


class ImportanceReuseSampler(ImportanceSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_seen = None

    def reset(self):
        super().reset()
        self.has_seen = None

    def set_weights(self, weights: np.ndarray):
        super().set_weights(weights)
        self.has_seen = np.repeat(False, len(self.weights))

    def sample(self, max_idx: int, s: int):
        if max_idx != len(self.weights):
            weights = self.weights[:max_idx]
            weights /= weights.sum()
        else:
            weights = self.weights


        rand_idxs = self.random.choice(max_idx, size=s*10, replace=True, p=weights)
        taken = 0
        self.sampled_idxs = []
        for rand_idx in rand_idxs:
            if not self.has_seen[rand_idx]:
                taken += 1
            self.has_seen[rand_idx] = True
            self.sampled_idxs.append(rand_idx)
            if taken >= s:
                break
        self.sampled_idxs = np.array(self.sampled_idxs)
        self.sampled_weights = self.weights[self.sampled_idxs]
        return self.sampled_idxs


class SamplingBounds:
    def __init__(self, delta):
        self.delta = delta

    # fx: bernoulli quantity
    def calc_bounds(self, fx):
        n = len(fx)

        mu = np.mean(fx)
        std = np.std(fx) / math.sqrt(n)
        k = math.sqrt(2 * math.log(1 / (2 * self.delta)))
        return mu - k * std, mu + k * std
