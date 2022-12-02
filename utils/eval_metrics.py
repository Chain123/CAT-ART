from sklearn import metrics
import torch
import os
import numpy as np
import _pickle as pickle


# classification  metrics
def Accuracy(logits, target):
    pred = torch.squeeze(logits) > 0
    equal = pred == torch.squeeze(target)
    acc = torch.sum(equal) / target.size()[0]
    return acc


def AUC(prob, target):
    fpr, tpr, thresholds = metrics.roc_curve(target, prob)
    return metrics.auc(fpr, tpr)


# recommendation metrics
def hit_at_k(r, k):
    """
    Args:
        r: rank of relevant item.
        k: Number of result to consider.
    Returns:
        hit at k
    """
    if r < k:
        return 1
    else:
        return 0


def hit_at_k_batch_old(r, k):
    hit = 0
    for rank in r:
        hit += hit_at_k(rank, k)
    return float(hit) / len(r)


def hit_at_k_batch(r, k):
    """
    :param r: list of rank, [Batch, 1] or [Batch]
    :param k: int, top K
    :return: Hit number for this batch.
    """
    r = (np.array(r) < k) + 0
    return float(sum(r)) / len(r)


def NDCG_at_k(r, k):
    """
    Args:
        r: rank of relevant item.
        k: Number of result to consider.
    Returns:
        NDCG at k
    """
    if r < k:
        return 1 / np.log2(r + 2)
    else:
        return 0


def NDCG_at_k_batch(r, k):
    ndcg = 0
    for rank in r:
        ndcg += NDCG_at_k(rank, k)
    return float(ndcg) / len(r)


def mrr_at_k(r, k):
    if r < k:
        return 1.0 / (r + 1)
    else:
        return 0.0


def mrr_at_k_batch(r, k):
    mrr = 0
    for rank in r:
        mrr += mrr_at_k(rank, k)
    return float(mrr) / len(r)


def metrics_rec(ranks, top_k):
    HR = []
    NDCG = []
    MRR = []
    for k_val in top_k:
        HR.append(hit_at_k_batch(ranks, k_val))
        NDCG.append(NDCG_at_k_batch(ranks, k_val))
        MRR.append(mrr_at_k_batch(ranks, k_val))
    return HR, NDCG, MRR


def print_all(data_dir):
    for file in os.listdir(data_dir):
        # print(file, os.path.join(data_dir, file, 'test_result.pickle'))
        if os.path.isfile(os.path.join(data_dir, file, 'test_result.pickle')):
            result = pickle.load(open(os.path.join(data_dir, file, 'test_result.pickle'), "rb"))
            print(file)
            for key in result.keys():
                print(key)
                for i in range(len(result[key])):
                    print(f"Sparse degree {i}")
                    print(result[key][i])

