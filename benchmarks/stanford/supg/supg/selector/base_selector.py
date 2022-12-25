from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource

class ApproxQuery:
    def __init__(
            self,
            qtype:str="pt",
            min_precision=None,
            min_recall=None,
            delta=0.01,
            budget=None,
    ):
        """
        :param type: pt, rt, prt
        :param min_precision:
        :param min_recall:
        :param delta:
        :param budget:
        """
        self.qtype = qtype
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.delta = delta
        self.budget = budget


class BaseSelector:
    def select(self) -> Sequence:
        raise NotImplemented
