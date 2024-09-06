#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 19:18
# @Author: zhangxiaotong
# @File  : Destination_Prediction_Module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNet(nn.Module):
    # The residual Networks to train easily and more robust.
    def __init__(self, input_size, num_final_fcs=4, hidden_size=128):
        super(ResidualNet, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        # 在这里声明了residual
        # self.residual = None

        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, inputs):
        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.relu(self.residuals[i](hidden))
            hidden = hidden + residual
        # 这里修改了self
        return residual


class Destination_Prediction_Moudle(nn.Module):
    def __init__(self, input_size, drop_prob=0):
        super(Destination_Prediction_Moudle, self).__init__()
        self.residual_net = ResidualNet(input_size)
        # 这里如果增加语义维度，需要把 2 改成 2+5=7
        # 这里试想一种新的思路，把out的经纬度和语义向量分开作为返回值
        self.hid2out_1 = nn.Linear(128, 2)
        self.hid2out_2 = nn.Linear(128, 5)
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, sptm_out):
        out = self.residual_net(sptm_out)
        active_out = torch.tanh(out)
        result_lnglat = self.hid2out_1(self.dropout(active_out))
        result_semantic = self.hid2out_2(self.dropout(active_out))
        return result_lnglat, result_semantic


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


class SemLoss(nn.Module):
    def __init__(self):
        super(SemLoss, self).__init__()

    def forward(self, pred, truth):
        """print(pred.shape)
        [32,4]
        print("\n")
        print(truth.shape)
        [32,5]
        """
        '''
        print("预测得到的向量：\n")
        print(pred)
        print("实际的向量：\n")
        print(truth)
        '''
        # pred = pred.contiguous().view(-1, 5)
        if pred.shape[1] == 5 and pred.shape[0] == 32:
            pred = pred.contiguous().view(-1, 5)
        else:
            pred = torch.zeros([32, 5], dtype=torch.float).view(-1, 5).cuda()
        if truth.shape[1] == 5 and truth.shape[0] == 32:
            truth = truth.contiguous().view(-1, 5)
        else:
            truth = torch.zeros([32, 5], dtype=torch.float).view(-1, 5).cuda()
        loss = get_similarity(pred, truth)
        # print("cos计算得到的loss:")
        # print(loss)
        return loss


def get_dis(pt0, pt1):
    # pt: [[lng, lat], ......], n x 2, torch.tensor
    #  change angle to radians
    RadPt0 = pt0 * math.pi / 180
    RadPt1 = pt1 * math.pi / 180
    delta = RadPt1 - RadPt0

    # a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    a = (delta[:, 1] / 2).sin() ** 2 + RadPt0[:, 1].cos() * RadPt1[:, 1].cos() * (delta[:, 0] / 2).sin() ** 2
    a = torch.clamp(a, min=0, max=1)
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371000
    return c * r


def get_similarity(vec0, vec1):
    cos_distance_Tensor = F.cosine_similarity(vec0, vec1, dim=0)
    similarity_Tensor = (cos_distance_Tensor.mul(-1)).add(1)
    return similarity_Tensor
