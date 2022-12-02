# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

l2_loss = nn.MSELoss(reduce=False)
cross_entropy_func = nn.CrossEntropyLoss()
cross_entropy_func_none = nn.CrossEntropyLoss(reduction='none')


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        assert temp > 0
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def l2_similarity(embed1, embed2, mask):
    """
    Args:
        embed1: shape [N, n_domain, latent_dim]
        embed2: shape [N, n_domain, latent_dim]
        mask: domain-level mask: [N, n_domain]
    return:
        l2 reconstruction loss between embedding a and b
    """
    loss = l2_loss(embed1, embed2)
    loss = loss.sum(axis=-1)
    if mask.sum() == 0:  # if no mask then reconstruction loss is meaningless.
        loss = (loss * mask).sum() / (mask.sum() + 1)
    else:
        loss = (loss * mask).sum() / mask.sum()
    return loss


def contrastive_loss_domain(embed_org, embed_aug, mask, sim, device):
    """
    Domain-level contrastive loss
    Args:
        embed_org (tensor [N, 5, latent_dim]), original user embedding in all 5 domains
        embed_aug (tensor [N, 5, latent_dim]), generated user embedding
                                               (masked domains are replaced with generated embedding)
        mask (tensor [N, 5]), 0-1 valued, where 1 represents the corresponding domain is masked.
        sim (Similarity), similarity function
        device
    """
    embed_org_splits = [torch.squeeze(val) for val in torch.split(embed_org, 1, dim=1)]
    embed_aug_splits = [torch.squeeze(val) for val in torch.split(embed_aug, 1, dim=1)]
    loss = 0
    for i in range(len(embed_aug_splits)):  # for each domain
        loss = loss + contrastive_loss_sample_N(embed_org_splits[i], embed_aug_splits[i], mask[:, i], sim, device)
    loss = loss / len(embed_aug_splits)
    return loss


def contrastive_loss_sample_2N(embed_org, embed_aug, mask, sim, device):
    """
    Sample-level contrastive loss
    Args:
        embed_org (tensor [N, latent_dim]), original user embedding in all 5 domains
        embed_aug (tensor [N, latent_dim]), generated user embedding
                                               (masked domains are replaced with generated embedding)
        mask (tensor [N, 5]), 0-1 valued, where 1 represents the corresponding domain is masked.
        sim (Similarity), similarity function
        device
    """
    # 1. concat embed_org and embed_aug --> embed_all
    embed_all = torch.cat((embed_org, embed_aug), dim=0)  # [2N, dim]
    # 2. cosine_similarity: mat_mul(embed_all, transpose(embed_all)) --> [bs * 2, bs*2]
    sim_scores = sim(embed_all.unsqueeze(1), embed_all.clone().unsqueeze(0))
    # 3. remove diagonal values pair(i,i)  (make it -1e9 or other equivalents)
    sim_scores.fill_diagonal_(-1e9)  # before cross-entropy
    # 4. ground truth pair: if index < batch_size: (index; index + batch_size) else: (index, index-batch_size)
    labels = torch.from_numpy((np.arange(embed_all.size(0)) + embed_org.size(0)) % embed_all.size(0)).long().to(device)
    # only sample and its mask -> generated sample are positive pairs.
    # 6. cross-entropy loss with logits, remove samples are have no mask
    loss = cross_entropy_func_none(sim_scores, labels)  # [2N]
    if len(mask.size()) > 1:  # 2D mask
        mask = (mask.sum(dim=1) > 0).to(float)
    else:
        mask = mask.to(float)

    if mask.sum() > 0:
        loss = (loss * mask).sum() / mask.sum()
    else:  # there is no mask in the whole batch.
        loss = 0
    return loss


def contrastive_loss_sample_N(embed_org, embed_aug, mask, sim, device='cuda'):
    """
    Sample-level contrastive loss
    Args:
        embed_org (tensor [N, latent_dim]), original user embedding in all 5 domains
        embed_aug (tensor [N, latent_dim]), generated user embedding
                                               (masked domains are replaced with generated embedding)
        mask (tensor [N, 5]), 0-1 valued, where 1 represents the corresponding domain is masked.
        sim (Similarity), similarity function
        device:
    """
    sim_scores_1 = sim(embed_org.unsqueeze(1), embed_aug.unsqueeze(0))
    sim_scores_2 = sim(embed_aug.unsqueeze(1), embed_org.unsqueeze(0))
    labels = torch.arange(embed_org.size(0)).long().to(device)
    # cross-entropy loss with logits, remove samples are have no mask
    loss_1 = cross_entropy_func_none(sim_scores_1, labels)  # [N]
    loss_2 = cross_entropy_func_none(sim_scores_2, labels)  # [N]
    loss = (loss_1 + loss_2) / 2  # positional averaging
    if len(mask.size()) > 1:  # 2D mask
        mask = (mask.sum(dim=1) > 0).to(float)
    else:
        mask = mask.to(float)
    if mask.sum() > 0:
        loss = (loss * mask).sum() / mask.sum()
    else:  # there is no mask in the whole batch.
        loss = 0
    return loss


########################
# Recommendation Metrics
########################
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def metrics_batch(batch_result):
    sorted_items = batch_result[0].numpy()
    groundTrue = batch_result[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in [5, 10, 20, 50]:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}
