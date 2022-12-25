import pandas as pd
import numpy as np
from scipy.special import logit, expit

from supg.datasource import DFDataSource


class BetaDataSource(DFDataSource):
    def __init__(
            self,
            alpha=0.01,
            beta=2.,
            N=1000000,
            seed=3212142,
            noise=None
    ):
        self.random = np.random.RandomState(seed)
        proxy_scores = self.random.beta(alpha, beta, size=N)
        true_labels = self.random.binomial(n=1, p=proxy_scores)
        print(sum(true_labels))

        if noise is not None:
            proxy_scores = proxy_scores + self.random.normal(scale=noise, size=len(proxy_scores))
            proxy_scores = proxy_scores.clip(0, 1)

        data = {'id': list(range(N)),
                'proxy_score': proxy_scores,
                'label': true_labels}
        df = pd.DataFrame(data)
        super().__init__(df)
