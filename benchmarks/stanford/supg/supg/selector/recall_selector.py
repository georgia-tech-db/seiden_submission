from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery


class RecallSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: Sampler,
        sample_mode: str = "sqrt",
        verbose: bool = False,
    ):
        self.query = query
        self.data = data
        self.sampler = sampler

        self.sample_mode = sample_mode
        self.verbose = verbose

    def log(self, str):
        if self.verbose:
            print(str)

    def select(self) -> Sequence:
        self.sampled = None

        data_idxs = self.data.get_ordered_idxs()
        budget = self.query.budget
        x_probs = self.data.get_y_prob()
        x_ranks = np.arange(len(x_probs))

        ## sqrt is the theoretically correct one
        if self.sample_mode == "prop":
            if not isinstance(self.sampler, ImportanceSampler):
                raise Exception("Invalid sampler for importance")
            self.sampler.set_weights(x_probs)
            sampler_weights = self.sampler.weights
        elif self.sample_mode == "uniform":
            if isinstance(self.sampler, ImportanceSampler):
                self.sampler.set_weights(np.repeat(1, len(x_probs)))
            sampler_weights = np.repeat(1, len(x_probs))
            sampler_weights = sampler_weights / np.sum(sampler_weights)
        else:
            if not isinstance(self.sampler, ImportanceSampler):
                raise Exception("Invalid sampler for importance")
            weights = np.sqrt(x_probs)
            self.sampler.set_weights(weights)
            sampler_weights = self.sampler.weights
        n_xs = len(data_idxs)
        x_ranks = np.arange(n_xs)
        x_weights = sampler_weights
        x_basep = np.repeat(1/n_xs, n_xs)
        x_masses = x_basep/x_weights

        s_ranks = np.sort(self.sampler.sample(max_idx=n_xs, s=budget))
        s_probs = x_probs[s_ranks]
        s_weights = x_weights[s_ranks]
        s_basep = x_basep[s_ranks]
        s_labels = self.data.lookup(data_idxs[s_ranks])
        s_masses = s_basep/s_weights

        # For joint
        self.sampled = np.unique(data_idxs[s_ranks])
        pos_sampled = self.data.filter(data_idxs[s_ranks])

        tot_pos_mass = np.sum(s_masses * s_labels)
        n_sample = budget
        n_pos = np.sum(s_labels)
        rt = self.query.min_recall
        target_mass = rt * tot_pos_mass
        cum_mass = 0
        t_s_idx = n_sample
        for i in range(n_sample):
            cum_mass += s_labels[i]*s_masses[i]
            if cum_mass >= target_mass:
                t_s_idx = i
                break
        t_u_idx = s_ranks[t_s_idx]
        self.log("t_s_idx: {} / {}. t_u_idx: {} / {}".format(
            t_s_idx, n_sample,
            t_u_idx, len(x_probs)
        ))

        s_idxs = np.arange(len(s_ranks))
        s_ind_l = s_idxs <= t_s_idx
        s_ind_r = s_idxs > t_s_idx
        u_ind_l = x_ranks <= t_u_idx
        u_ind_r = x_ranks > t_u_idx

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / 2)
        _, s_left_ub = bounder.calc_bounds(
            fx=s_labels * s_masses * s_ind_l,
        )
        s_right_lb, _ = bounder.calc_bounds(
            fx=s_labels * s_masses * s_ind_r,
        )

        self.log("left_adj: {}, right_adj: {}".format(s_left_ub, s_right_lb))
        rc = s_left_ub / (s_left_ub + s_right_lb)
        self.log("Rc: {}".format(rc))

        if rc >= 1.:
            return np.array(list(range(n_xs)))

        t_adj_s_idx = n_sample - 1
        cum_mass = 0
        s_pos_idxs = []
        for i in range(n_sample):
            if s_labels[i]:
                cum_mass += s_masses[i]
                s_pos_idxs.append(i)
            if cum_mass >= rc * tot_pos_mass:
                t_adj_s_idx = i
                break
        t_adj_u_idx = s_ranks[t_adj_s_idx]

        set_ids = data_idxs[:t_adj_u_idx+1]
        all_inds = np.unique(
                np.concatenate([set_ids, pos_sampled])
        )
        return all_inds
