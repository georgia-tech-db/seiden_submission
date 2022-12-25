from collections import defaultdict

import pandas as pd
import numpy as np

import supg.datasource as datasource
import supg.selector as selector
import supg.sampler as sampler

from tqdm import tqdm


class TrialRunner:
    def __init__(self):
        pass

    def run_trials(
            self,
            selector: selector.BaseSelector,
            query: selector.ApproxQuery,
            sampler: sampler.Sampler,
            source: datasource.DataSource,
            nb_trials: int = 100,
            verbose: bool = True,
    ):
        results = []
        ordered_ids = source.get_ordered_idxs()
        true_labels = source.lookup(ordered_ids)
        nb_true = np.sum(true_labels)

        if verbose:
            itr = tqdm(range(nb_trials))
        else:
            itr = range(nb_trials)
        for i in itr:
            inds = selector.select()
            nb_got = np.sum(source.lookup(inds))
            prec = nb_got / len(inds)
            recall = nb_got / nb_true
            if query.qtype == 'jt':
                nb_sampled = selector.total_sampled
            else:
                nb_sampled = query.budget
            results.append({
                "query_type": query.qtype,
                "precision": prec,
                "recall": recall,
                'size': len(inds),
                'nb_true': nb_true,
                'nb_sampled': nb_sampled,
                "trial_idx": i,
            })
            sampler.reset()

        results_df = pd.DataFrame(results)
        if query.qtype == "pt":
            results_df["covered"] = results_df["precision"] > query.min_precision
        elif query.qtype == "rt":
            results_df["covered"] = results_df["recall"] > query.min_recall
        elif query.qtype == "jt":
            results_df["covered"] = (
                (results_df["recall"] > query.min_recall)
                & (results_df["precision"] > query.min_precision)
            )
        else:
            results_df["covered"] = False
        # print("Frac Correct: {}".format(np.mean(results_df["covered"])))
        # print("Avg Recall: {}".format(np.mean(results_df["recall"])))
        return results_df
