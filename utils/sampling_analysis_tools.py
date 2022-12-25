"""

In this file, we will write functions to fully visualize and analyze the performance of our clustering methods


"""
import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def cluster_stats(all_cluster_labels, draw = False):
    """
    In this function, we want to derive the mean and variance of cluster as the number of cluster differs
    :param all_cluster_labels:
    :return:
    """

    cluster_count = np.zeros(max(all_cluster_labels) + 1)
    dictionary = dict()
    dictionary['size'] = cluster_count

    for cluster_label in all_cluster_labels:
        cluster_count[cluster_label] += 1

    print(f"Median: {np.median(cluster_count)}, Variance: {np.var(cluster_count)}")
    print(f"Normalized Median: {np.median(cluster_count) / len(cluster_count)}, "
          f"Normalized Var: {np.var(cluster_count) / len(cluster_count)}")

    if draw:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        x_axis = [i for i in range(len(cluster_count))]
        ## let's sort the y_axis by cluster_count
        cluster_count = np.sort(cluster_count)
        y_axis = cluster_count
        ax.bar(x_axis, y_axis)
        plt.show()


### to understand the performance of our algorithms, we need to understand what examples gave us failures.
# For sampling, we are all dealing with representative frames / mappings,
# we will use this information to derive the information we need
def examine_wrong_examples(propagated_labels, gt_labels, rep_indices, mapping):
    """

    :param propagated_labels: predicted labels
    :param gt_labels: ground truth labels
    :param rep_indices: indices from the original array used to choose representative frames
    :param mapping: indexes that refer to the rep_frame / rep_label
    :return:
    """

    ## can we convert this mapping to the indexes in the original array?

    wrong_examples = {}
    for i in range(len(propagated_labels)):
        if propagated_labels[i] != gt_labels[i]:
            try:
                wrong_examples[i] = [rep_indices[mapping[i]], propagated_labels[i], gt_labels[i], propagated_labels[mapping[i]], gt_labels[mapping[i]]]
            except:
                print(f"failed, {i}, {mapping[i]}, {len(rep_indices)}")
            # order is [index_of_rep_frame, it's proposed_label, it's gt label, the rep frame proposed_label (should be same as it's proposed label), rep frame gt_label]

    return wrong_examples


def accuracy_by_cluster(predicted_labels, gt_labels, all_cluster_labels):
    """
    We want to see the correlation between cluster size and query_accuracy

    :param predicted_labels: prediction generated through sampling
    :param gt_labels: ground truth labels
    :param all_cluster_labels: cluster assignments
    :return:
    """
    ## steps: 1. determine the cluster set
    cluster_count = np.zeros(max(all_cluster_labels) + 1)
    dictionary = dict()
    dictionary['size'] = cluster_count

    for cluster_label in all_cluster_labels:
        cluster_count[cluster_label] += 1
    for key in gt_labels.keys():
        dictionary[key] = []
        pred = predicted_labels[key]
        gt = gt_labels[key]
        truth_arr = pred == gt
        true_counts = np.zeros(max(all_cluster_labels) + 1)
        for i, truth_val in enumerate(truth_arr):
            if truth_val:
                true_counts[all_cluster_labels[i]] += 1
        dictionary[key] = true_counts / cluster_count ## we save the accuracies for each cluster

    df = pd.DataFrame(dictionary, columns = dictionary.keys())
    return df



def f1_by_cluster(predicted_labels, gt_labels, all_cluster_labels):
    """
    We want to see the correlation between cluster size and query_accuracy

    :param predicted_labels: prediction generated through sampling
    :param gt_labels: ground truth labels
    :param all_cluster_labels: cluster assignments
    :return:
    """
    ## steps: 1. determine the cluster set
    cluster_count = np.zeros(max(all_cluster_labels) + 1)
    dictionary = dict()
    dictionary['size'] = cluster_count

    ## how many members are there in each cluster?
    for cluster_label in all_cluster_labels:
        cluster_count[cluster_label] += 1

    ## we know the max cluster num, min cluster num and everything in between.
    for key in gt_labels.keys():
        pred = predicted_labels[key]
        gt = gt_labels[key]
        pred = np.array(pred, dtype = np.int)
        gt = np.array(gt, dtype = np.int)

        precisions = []
        recalls = []
        f1_scores = []
        non_tp_count = 0
        print('======================================')
        for cluster_num in range(len(cluster_count)):


            pred_cluster_i = pred[all_cluster_labels == cluster_num]
            gt_cluster_i = gt[all_cluster_labels == cluster_num]

            tp_arr = np.logical_and(pred_cluster_i, gt_cluster_i)
            fn_arr = np.logical_and(np.logical_not(pred_cluster_i), gt_cluster_i)
            fp_arr = np.logical_and(pred_cluster_i, np.logical_not(gt_cluster_i))
            if not (sum(tp_arr) == 0 and sum(fp_arr) == 0 and sum(fn_arr) == 0):
                print(f"[{cluster_num}] size: {len(tp_arr)} predicted 1: {sum(pred_cluster_i)}, "
                      f"actual 1: {sum(gt_cluster_i)}, tp: {sum(tp_arr)} ({sum(tp_arr) / len(tp_arr)}), "
                      f"fp: {sum(fp_arr)} ({sum(fp_arr) / len(fp_arr)}), "
                      f"fn: {sum(fn_arr)} ({sum(fn_arr) / len(fn_arr)})")

            if sum(tp_arr) == 0 and sum(fn_arr) == 0 and sum(fp_arr) == 0:
                #print(f"cluster num: {cluster_num} does not have tp")

                precisions.append(-1)
                recalls.append(-1)
                f1_scores.append(-1)

            elif sum(tp_arr) == 0 and sum(fp_arr) == 0:
                precisions.append(-1)
                recalls.append(metrics.recall_score(gt_cluster_i, pred_cluster_i))
                f1_scores.append(metrics.f1_score(gt_cluster_i, pred_cluster_i))

            elif sum(tp_arr) == 0 and sum(fn_arr) == 0:
                precisions.append(metrics.precision_score(gt_cluster_i, pred_cluster_i))
                recalls.append(-1)
                f1_scores.append(metrics.f1_score(gt_cluster_i, pred_cluster_i))

            else:
                #print(f"{cluster_num} Sum of tp_arr is {sum(tp_arr)}")
                precisions.append(metrics.precision_score(gt_cluster_i, pred_cluster_i))
                recalls.append(metrics.recall_score(gt_cluster_i, pred_cluster_i))
                f1_scores.append(metrics.f1_score(gt_cluster_i, pred_cluster_i))
        print('==========================================')
        print(f"Non tp count is {non_tp_count}")
        truth_arr = pred == gt
        true_counts = np.zeros(max(all_cluster_labels) + 1)
        for i, truth_val in enumerate(truth_arr):
            if truth_val:
                true_counts[all_cluster_labels[i]] += 1
        dictionary[key + 'precision'] = precisions
        dictionary[key + 'recall'] = recalls  ## we save the accuracies for each cluster
        dictionary[key + 'f1'] = f1_scores

    df = pd.DataFrame(dictionary, columns = dictionary.keys())
    return df


