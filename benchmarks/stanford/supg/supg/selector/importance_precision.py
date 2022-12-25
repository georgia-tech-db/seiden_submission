from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery


class ImportancePrecisionSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: ImportanceSampler,
        start_samp=100,
        step_size=100
    ):
        self.query = query
        self.data = data
        self.sampler = sampler
        if not isinstance(sampler, ImportanceSampler):
            raise Exception("Invalid sampler for importance")
        self.start_samp = start_samp
        self.step_size = step_size

    def select(self) -> Sequence:
        data_idxs = self.data.get_ordered_idxs()
        n = len(data_idxs)
        T = 1 + 2 * (self.query.budget - self.start_samp) // self.step_size
        # TODO: weights
        x_probs = self.data.get_y_prob()
        weights = np.sqrt(x_probs)
        # self.sampler.set_weights(np.repeat(1.,n)/n)
        self.sampler.set_weights(weights)
        # self.sampler.set_weights(x_probs ** 2)
        # self.sampler.set_weights(x_probs)

        x_ranks = np.arange(n)
        x_basep = np.repeat((1./n),n)
        x_weights = self.sampler.weights

        samp_ranks = np.sort(self.sampler.sample(max_idx=n, s=self.query.budget))
        n_samp = len(samp_ranks)
        samp_ids = data_idxs[samp_ranks]
        samp_labels = self.data.lookup(samp_ids)

        delta = self.query.delta
        allowed = [0]
        for s_idx in range(self.start_samp, n_samp, self.step_size):
            cur_u_idx = samp_ranks[s_idx]
            cur_x_basep = x_basep[:cur_u_idx+1] / np.sum(x_basep[:cur_u_idx+1])
            cur_x_weights = x_weights[:cur_u_idx+1] / np.sum(x_weights[:cur_u_idx+1])
            cur_x_masses = cur_x_basep / cur_x_weights

            cur_subsample_x_idxs = samp_ranks[:s_idx+1]

            bounder = SamplingBounds(delta=delta / T)
            pos_rank_lb, pos_rank_ub = bounder.calc_bounds(
                fx=samp_labels[:s_idx+1]*cur_x_masses[cur_subsample_x_idxs],
            )
            prec_lb = pos_rank_lb
            if prec_lb > self.query.min_precision:
                allowed.append(cur_u_idx)

        set_inds = data_idxs[:allowed[-1]]
        samp_inds = self.data.filter(samp_ids)
        return np.unique(np.concatenate([set_inds, samp_inds]))
