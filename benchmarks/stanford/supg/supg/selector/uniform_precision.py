from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery
from .naive_recall import calc_lb


class UniformPrecisionSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: Sampler,
        start_samp=100,
        step_size=100
    ):
        self.query = query
        self.data = data
        self.sampler = sampler
        self.start_samp = start_samp
        self.step_size = step_size

    def select(self) -> Sequence:
        data_idxs = self.data.get_ordered_idxs()
        n = len(data_idxs)
        T = 1 + (self.query.budget - self.start_samp) // self.step_size
        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.set_weights(np.repeat(1, n))

        samp_ranks = self.sampler.sample(max_idx=n, s=self.query.budget)
        samp_ids = data_idxs[samp_ranks]

        true = self.data.lookup(samp_ids)
        proxy = self.data.lookup_yprob(samp_ids)
        ordered = sorted(list(
                zip(proxy, true, list(range(self.query.budget)))), reverse=True)
        ordered = list(ordered)

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / T)
        allowed = [0]
        for rank in range(self.start_samp, self.query.budget, self.step_size):
            trues = np.array([x[1] for x in ordered[0:rank]])
            _, prec_lb = bounder.calc_bounds(
                fx=trues
            )
            if prec_lb > self.query.min_precision:
                allowed.append(rank)
        if allowed[-1] == 0:
            return_rank = 0
        else:
            return_rank = samp_ranks[ordered[allowed[-1]][2]]

        set_inds = data_idxs[:return_rank]
        samp_inds = self.data.filter(samp_ids)
        return np.unique(np.concatenate([set_inds, samp_inds]))
