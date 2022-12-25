from typing import Sequence

import numpy as np

from supg.selector import RecallSelector


class JointSelector(RecallSelector):
    def select(self) -> Sequence:
        all_inds = super().select()
        sampled = self.sampled

        left = list(set(all_inds) - set(sampled))
        left = np.array(left)

        self.total_sampled = len(left) + self.query.budget

        return self.data.filter(all_inds)
