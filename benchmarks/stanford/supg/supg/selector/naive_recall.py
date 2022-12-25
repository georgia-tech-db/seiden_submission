from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler
from supg.selector.base_selector import BaseSelector, ApproxQuery


def calc_lb(p, n, delta, T):
    # Lower bound on the precision of a sample with failure probability delta
    # not exactly right but has the right scaling
    return p - math.sqrt(p*(1-p)/n)*math.sqrt(2*math.log(T/delta))


class NaiveRecallSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: Sampler
    ):
        self.query = query
        self.data = data
        self.sampler = sampler

    def select(self) -> Sequence:
        data_idxs = self.data.get_ordered_idxs()
        n = len(data_idxs)
        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.set_weights(np.repeat(1, n))

        samp_ranks = self.sampler.sample(max_idx=n, s=self.query.budget)
        samp_ids = data_idxs[samp_ranks]

        true = self.data.lookup(samp_ids)
        proxy = self.data.lookup_yprob(samp_ids)
        ordered = sorted(list(
                zip(proxy, true, list(range(self.query.budget)))), reverse=True)
        ordered = list(ordered)

        allowed = [-1]
        nb_true = 0.
        nb_tot = np.sum(true)
        for rank in range(self.query.budget):
            nb_true += ordered[rank][1]
            recall_est = nb_true / nb_tot
            if recall_est > self.query.min_recall:
                allowed.append(rank)
                break
        return_rank = samp_ranks[ordered[allowed[-1]][2]]

        set_inds = data_idxs[:return_rank]
        samp_inds = self.data.filter(samp_ids)
        return np.unique(np.concatenate([set_inds, samp_inds]))
