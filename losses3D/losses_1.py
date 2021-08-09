# -*- coding:utf-8 -*-
import torch as t
from torch import nn
import torch.nn.functional as F
from math import exp

import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-3

    def dsc(self, y_pred, y_true):
        batch_size = y_true.size(0)
        m1 = y_true.view(batch_size, -1)
        m2 = y_pred.view(batch_size, -1)
        intersection = m1 * m2
        score = (2. * intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = score.sum() / batch_size
        return score

    def forward(self, y_pred, y_true):
        loss = 1 - self.dsc(y_pred, y_true)
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = nn.BCELoss()(y_pred, y_true)
        return dice_loss + bce_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 假阳性的惩罚系数
        self.beta = beta  # 假阴性的惩罚系数
        self.smooth = 1e-3

    def tversky(self, y_pred, y_true):
        batch_size = y_true.size(0)
        m1 = y_true.view(batch_size, -1)
        m2 = y_pred.view(batch_size, -1)
        # intersection = m1 * m2
        true_pos = m1 * m2
        false_pos = m2 * (1 - m1)
        false_neg = m1 * (1 - m2)
        score = (true_pos.sum(1) + self.smooth) / (
                    true_pos.sum(1) + self.alpha * false_pos.sum(1) + self.beta * false_neg.sum(1) + self.smooth)
        return score.sum() / batch_size

    def forward(self, y_pred, y_true):
        loss = 1 - self.tversky(y_pred, y_true)
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, gamma=0.75, alpha=0.3, beta=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky_loss = TverskyLoss(self.alpha, self.beta)

    def forward(self, y_pred, y_true):
        pt_1 = self.tversky_loss(y_pred, y_true)
        return t.pow((1 - pt_1), self.gamma)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)
        pt = t.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return t.mean(F_loss)
        else:
            return F_loss


if __name__ == "__main__":
    y_true = t.randn((5, 1, 55, 55))
    y_pred = t.randn((5, 1, 55, 55))
    print(DiceLoss()(y_true, y_pred))
    print(TverskyLoss(alpha=0.1, beta=0.2)(y_true, y_pred))
    print(FocalTverskyLoss(gamma=0.72, alpha=0.1, beta=0.2)(y_true, y_pred))



