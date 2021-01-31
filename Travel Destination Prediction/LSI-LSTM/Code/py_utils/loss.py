import math
import torch
import torch.nn as nn


def get_dis(pt0, pt1):
    RadPt0 = pt0 * math.pi / 180
    RadPt1 = pt1 * math.pi / 180
    delta = RadPt1 - RadPt0

    a = (delta[:, 1] / 2).sin() ** 2 + RadPt0[:, 1].cos() * RadPt1[:, 1].cos() * (delta[:, 0] / 2).sin() ** 2
    a = torch.clamp(a, min=0, max=1)
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371000
    return c * r


class DisLoss(nn.Module):
    def __init__(self, is_MAE=False):
        super(DisLoss, self).__init__()
        self.is_MAE = is_MAE
        # get a batch of losses
        self.Loss = nn.L1Loss(reduction='none')

    def forward(self, pred, truth):
        pred = pred.contiguous().view(-1, 2)
        truth = truth.contiguous().view(-1, 2)
        if self.is_MAE:
            loss = self.Loss(pred, truth)
        else:
            loss = get_dis(pred, truth)
        return loss