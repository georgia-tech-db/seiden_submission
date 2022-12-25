import math
import numpy as np

from .base_estimator import Estimator


class NaiveEstimator(Estimator):
    def estimate(self, proxy_scores, true_labels, max_idx, delta, T, return_upper):
        mean = np.mean(true_labels)
        ci = self.calc_bernoulli_ci(mean, len(true_labels), delta, T)
        if return_upper:
            return mean + ci
        else:
            return mean - ci
