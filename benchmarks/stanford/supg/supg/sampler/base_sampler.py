from typing import Optional, Sequence
from collections import namedtuple


SampleRange = namedtuple('SampleRange', ['l', 'r', 's'])

class Sampler:
    def sample(self, max_idx: int, s: int) -> Sequence[int]:
        raise NotImplemented()

    def reset(self):
        raise NotImplemented()

    def get_distinct_samples(self):
        raise NotImplemented()
