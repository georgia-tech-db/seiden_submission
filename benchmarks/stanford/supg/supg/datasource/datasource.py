from typing import List, Sequence

import pandas as pd
import numpy as np


class DataSource:
    def lookup(self, idxs: Sequence) -> np.ndarray:
        raise NotImplemented()

    def filter(self, ids) -> np.ndarray:
        labels = self.lookup(ids)
        return np.array([ids[i] for i in range(len(ids)) if labels[i]])

    def get_ordered_idxs(self) -> np.ndarray:
        raise NotImplemented()

    def get_y_prob(self) -> np.ndarray:
        raise NotImplemented()

    def lookup_yprob(self, ids) -> np.ndarray:
        raise NotImplemented()


class RealtimeDataSource(DataSource):
    def __init__(
        self,
        y_pred,
        y_true,
        seed=123041,
    ):
        self.y_pred = y_pred
        self.y_true = y_true
        self.random = np.random.RandomState(seed)
        self.proxy_score_sort = np.lexsort((self.random.random(y_pred.size), y_pred))[::-1]
        self.lookups = 0

    def lookup(self, ids):
        self.lookups += len(ids)
        return self.y_true[ids]

    def get_ordered_idxs(self) -> np.ndarray:
        return self.proxy_score_sort

    def get_y_prob(self) -> np.ndarray:
        return self.y_pred[self.proxy_score_sort]

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.y_pred[ids]


class DFDataSource(DataSource):
    def __init__(
            self,
            df,
            drop_p=None,
            seed=123041
    ):
        self.random = np.random.RandomState(seed)
        if drop_p is not None:
            pos = df[df['label'] == 1]
            remove_n = int(len(pos) * drop_p)
            drop_indices = self.random.choice(pos.index, remove_n, replace=False)
            df = df.drop(drop_indices).reset_index(drop=True)
            df.id = df.index

        print(len(df[df['label'] == 1]) / len(df))
        self.df_indexed = df.set_index(["id"])
        self.df_sorted = df.sort_values(
                ["proxy_score"], axis=0, ascending=False).reset_index(drop=True)
        self.lookups = 0

    def lookup(self, ids):
        self.lookups += len(ids)
        return self.df_indexed.loc[ids]["label"].values

    def get_ordered_idxs(self) -> np.ndarray:
        return self.df_sorted["id"].values

    def get_y_prob(self) -> np.ndarray:
        return self.df_sorted["proxy_score"].values

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.df_indexed.loc[ids]['proxy_score'].values
