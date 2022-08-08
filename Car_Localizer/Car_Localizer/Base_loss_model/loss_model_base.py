import torch
import torch.nn as nn
import torch.nn.functional as F


class LossBase(nn.Module):
    def __init__(self):
        super(LossBase, self).__init__()

        self.Alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Sigma = nn.Parameter(torch.ones(1), requires_grad=True)

    def my_bce_with_logits_loss(x, y):
        loss = -1.0 * (y * F.logsigmoid(x) + (1 - y) * torch.log(1 - torch.sigmoid(x)))
        loss = loss.mean()
        return loss

    def my_bce_with_logits_loss_stable(x, y):
        max_val = (-x).clamp_min_(0)
        loss = (
            (1 - y) * x
            + max_val
            + torch.log(torch.exp(-max_val) + torch.exp(-x - max_val))
        )
        loss = loss.mean()
        return loss
