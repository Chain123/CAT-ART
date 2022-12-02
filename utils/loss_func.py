import torch
import torch.nn as nn


def BPRLoss(pos_scores, neg_scores, show=None):
    pos_scores = torch.squeeze(pos_scores)
    if len(neg_scores.size()) == 2:  # one pos vs multiple neg
        score_b = torch.mean(neg_scores, 1)
    # diff = torch.squeeze(score_a) - torch.squeeze(score_b)
    loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
    # remove large val for stability of sigmoid function.
    # diff[diff > 40] = 40
    # diff[diff < -40] = -40
    # # mask positions where show = 0, means that this user never show up in corresponding domain.
    # if show is not None:
    #     diff[show == 0] = 0
    # return -diff.sigmoid().log().mean()
    return loss

# def cross_entropy(logits, targets):
#     loss = torch.nn.CrossEntropyLoss()
