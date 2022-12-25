from typing import Sequence

import math
import numpy as np

class Estimator:
    def estimate(
            self,
            proxy_scores: Sequence[float],
            true_labels: Sequence[float],
            max_idx: int,
            delta: float,
            T: int,
            return_upper: bool,
    ) -> float:
        raise NotImplemented()

    def estimate_ub(self, proxy_scores, true_labels, max_idx, delta, T):
        return self.estimate(proxy_scores, true_labels, max_idx, delta, T, True)

    def estimate_lb(self, proxy_scores, true_labels, max_idx, delta, T):
        return self.estimate(proxy_scores, true_labels, max_idx, delta, T, False)

    # One sided CI
    def calc_bernoulli_ci(self, p, n, delta, T):
        # Lower bound on the precision of a sample with failure probability delta
        # not exactly right but has the right scaling
        return math.sqrt(p * (1-p) / n) * \
            math.sqrt(2 * math.log(T / delta))

    # One sided CI
    def calc_std_ci(self, samples, delta, T):
        std = np.std(samples)
        return std / np.sqrt(len(samples)) * \
            np.sqrt(2 * math.log(T / delta))
