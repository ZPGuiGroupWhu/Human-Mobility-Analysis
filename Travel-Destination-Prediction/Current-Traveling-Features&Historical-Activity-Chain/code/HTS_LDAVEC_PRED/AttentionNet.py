#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 19:56
# @Author: zhangxiaotong
# @File  : AttentionNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# from Model_new.config_parameter import config


class Scorer(nn.Module):
    def forward(self, traj):
        # spd = (1 / (traj['spd'] + 1.0)).unsqueeze(-1)
        # v_mean = (1 / (traj['v_mean'] + 1.0)).unsqueeze(-1)
        # v_std = traj['v_std'].unsqueeze(-1)
        # angle = (traj['angle'] + 1.0).unsqueeze(-1)
        dis = traj['travel_dis'].unsqueeze(-1)
        angle = traj['angle'].unsqueeze(-1)
        # c_entropy = traj['c_entropy'].unsqueeze(-1)
        # d_entropy = traj['d_entropy'].unsqueeze(-1)
        # his_semantic = recently['history_semantic_0'].unsqueeze(-1)
        # print(his_semantic.shape)
        # 这里加了一个转向角的注意力
        score = torch.cat((dis, angle), dim=2)
        # print(score.shape)
        # history_score = torch.cat((his_semantic, his_semantic, his_semantic, his_semantic), dim=2)
        # score = torch.cat((score, history_score), dim = 1)
        # print("注意力的维度：\n")
        # print(score.shape)
        return score


class History_Scorer(nn.Module):
    def __init__(self, config):
        super(History_Scorer, self).__init__()
        self.config = config

    def forward(self, history, traj):
        # spd = (1 / (traj['spd'] + 1.0)).unsqueeze(-1)
        # v_mean = (1 / (traj['v_mean'] + 1.0)).unsqueeze(-1)
        # v_std = traj['v_std'].unsqueeze(-1)
        # angle = (traj['angle'] + 1.0).unsqueeze(-1)
        # dis = traj['travel_dis'].unsqueeze(-1)
        # angle = traj['angle'].unsqueeze(-1)
        # c_entropy = traj['c_entropy'].unsqueeze(-1)
        # d_entropy = traj['d_entropy'].unsqueeze(-1)
        # 之前的注意力系数权重为第一维地理语义，这里修改为时间注意力
        # his_semantic = recently['history_semantic_0'].unsqueeze(-1)
        weekday_s = traj['weekday_s']
        weekday_c = traj['weekday_c']
        start_time_s = traj['start_time_s']
        start_time_c = traj['start_time_c']
        # 获取当前的注意力参考值，日期的正余弦，时间的正余弦
        current_weekday_sin = weekday_s.cpu().numpy()[0][0]
        current_weekday_cos = weekday_c.cpu().numpy()[0][0]
        # print(current_weekday_sin)   要使得输出的值为一个标量常数，不能是一个列表或二维数组
        current_start_time_sin = start_time_s.cpu().numpy()[0][0]
        current_start_time_cos = start_time_c.cpu().numpy()[0][0]
        # Tensor每一行减去一个常数得到的差值结果
        # print(recently)
        # 输出历史出行活动链
        '''history_weekday_sin = history['history_weekday_sin'].unsqueeze(-1)[:, :-config['cp_lens'],
                              :] - current_weekday_sin
        history_weekday_cos = history['history_weekday_cos'].unsqueeze(-1)[:, :-config['cp_lens'],
                              :] - current_weekday_cos
        history_start_time_sin = history['history_start_time_sin'].unsqueeze(-1)[:, :-config['cp_lens'],
                                 :] - current_start_time_sin
        history_start_time_cos = history['history_start_time_cos'].unsqueeze(-1)[:, :-config['cp_lens'],
                                 :] - current_start_time_cos'''
        history_weekday_sin = 1.0 / (
            (torch.abs(history['history_weekday_sin'].unsqueeze(-1)[:, :-self.config['cp_lens'],
                       :] - current_weekday_sin)).add(0.001))
        history_weekday_cos = 1.0 / (
            (torch.abs(history['history_weekday_cos'].unsqueeze(-1)[:, :-self.config['cp_lens'],
                       :] - current_weekday_cos)).add(0.001))
        history_start_time_sin = 1.0 / (
            (torch.abs(history['history_start_time_sin'].unsqueeze(-1)[:, :-self.config['cp_lens'],
                       :] - current_start_time_sin)).add(0.001))
        history_start_time_cos = 1.0 / (
            (torch.abs(history['history_start_time_cos'].unsqueeze(-1)[:, :-self.config['cp_lens'],
                       :] - current_start_time_cos)).add(0.001))
        his_score = torch.cat((history_weekday_sin, history_weekday_cos, history_weekday_sin, history_weekday_cos,
                               history_start_time_sin, history_start_time_cos), dim=2)
        # 这里测试直接把历史的一维作为注意力相乘
        # 这里加了一个转向角的注意力
        # score = torch.cat((dis,c_entropy,d_entropy,angle), dim=2)
        # print(score.shape)
        # history_score = torch.cat((his_semantic, his_semantic, his_semantic, his_semantic), dim=2)
        # score = torch.cat((score, history_score), dim = 1)
        # print("注意力的维度：\n")
        # print(his_semantic.shape)
        # print(his_score)      ——本身
        # print(his_score.shape)        ——大小
        return his_score


class Recently_Score(nn.Module):
    def __init__(self, config):
        super(Recently_Score, self).__init__()
        self.config = config

    def forward(self, recently, traj):
        weekday_s = traj['weekday_s']
        weekday_c = traj['weekday_c']
        start_time_s = traj['start_time_s']
        start_time_c = traj['start_time_c']
        lng_c = traj['lng']
        lat_c = traj['lat']
        # 获取当前的注意力参考值，日期的正余弦，时间的正余弦
        current_weekday_sin = weekday_s.cpu().numpy()[0][0]
        current_weekday_cos = weekday_c.cpu().numpy()[0][0]
        # print(current_weekday_sin)   要使得输出的值为一个标量常数，不能是一个列表或二维数组
        current_start_time_sin = start_time_s.cpu().numpy()[0][0]
        current_start_time_cos = start_time_c.cpu().numpy()[0][0]
        current_lng = lng_c.cpu().numpy()[0][0]
        current_lat = lat_c.cpu().numpy()[0][0]
        # Tensor每一行减去一个常数得到的差值结果
        # print(recently)
        # 输出历史出行活动链
        history_weekday_sin = 1.0 / (
            (torch.abs(recently['history_weekday_sin'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_weekday_sin)).add(0.001))
        history_weekday_cos = 1.0 / (
            (torch.abs(recently['history_weekday_cos'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_weekday_cos)).add(0.001))
        history_start_time_sin = 1.0 / (
            (torch.abs(recently['history_start_time_sin'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_start_time_sin)).add(0.001))
        history_start_time_cos = 1.0 / (
            (torch.abs(recently['history_start_time_cos'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_start_time_cos)).add(0.001))
        history_lng = 1.0 / (
            (torch.abs(recently['history_lng'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_lng)).add(0.001))
        history_lat = 1.0 / (
            (torch.abs(recently['history_lat'].unsqueeze(-1)[:, -self.config['cp_lens']:,
                       :] - current_lng)).add(0.001))
        recent_score = torch.cat((history_lng, history_lat, history_weekday_sin, history_weekday_cos,
                                  history_start_time_sin, history_start_time_cos), dim=2)
        return recent_score


class Spatial_AttentionNet(nn.Module):

    def __init__(self, config):
        super(Spatial_AttentionNet, self).__init__()
        self.config = config
        # attn 1
        self.projector = nn.Sequential(
            nn.Linear(2, self.config['hidden_size']),  # 这里的128和hidden_size一致
            nn.ReLU()
        )

        # # attn 2  #
        # self.projector = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.Tanh()
        # )

    def forward(self, hidden, score):
        # attn_solve1 #
        # (B x L x H) = (B x L x 3)*(3, H)
        # print('self.projector(score):',self.projector(score).size())
        # print('hidden:',hidden.size())
        # print("score.shape:")
        # print(score.shape)
        weights = F.softmax(self.projector(score), dim=1)
        # weights = F.softmax(torch.mul(self.projector(score), hidden), dim=1)
        # (B x H) = (B x L x H)*(B x L x H).sum(dim=1)
        # weights = torch.cat((weights,weights),dim=1)
        atten_result = (hidden * weights).sum(dim=1)
        # atten_result = hidden * weights atten_solve2  The paper On Prediction of User Destination by Sub-Trajectory
        # Understanding: A Deep Learning based Approach # weights = F.softmax(self.projector(hidden),
        # dim=1) atten_result = hidden * weights print("hidden,atten_result:",hidden.size(),atten_result.size())
        return atten_result, weights


class History_AttentionNet(nn.Module):

    def __init__(self, config):
        super(History_AttentionNet, self).__init__()
        self.config = config
        # atten 1
        self.projector = nn.Sequential(
            nn.Linear(6, self.config['hist_hidden_size']),  # 这里的128和hidden_size一致
            nn.ReLU()
        )

        # # atten 2  #
        # self.projector = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.Tanh()
        # )

    def forward(self, hidden, score):
        # atten_solve1 #
        # (B x L x H) = (B x L x 3)*(3, H)
        # print('self.projector(score):',self.projector(score).size())
        # print('hidden:',hidden.size())
        # print("score.shape:")
        # print(score.shape)
        weights = F.softmax(self.projector(score), dim=1)
        # weights = F.softmax(torch.mul(self.projector(score), hidden), dim=1)
        # (B x H) = (B x L x H)*(B x L x H).sum(dim=1)
        # weights = torch.cat((weights,weights),dim=1)
        atten_result = (hidden * weights).sum(dim=1)
        # atten_result = hidden * weights atten_solve2  The paper On Prediction of User Destination by Sub-Trajectory
        # Understanding: A Deep Learning based Approach # weights = F.softmax(self.projector(hidden),
        # dim=1) atten_result = hidden * weights print("hidden,atten_result:",hidden.size(),atten_result.size())
        return atten_result, weights
