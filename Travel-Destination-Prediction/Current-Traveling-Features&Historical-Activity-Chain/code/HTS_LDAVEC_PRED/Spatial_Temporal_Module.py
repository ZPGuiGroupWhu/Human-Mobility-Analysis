#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 19:49
# @Author: zhangxiaotong
# @File  : Spatial_Temporal_Module.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Model_new.HTS_LDAVEC_PRED import AttentionNet


class TemporalNet(nn.Module):
    """
    Net to get the temporal information of the trajectory_seqs by using RNN
    """

    def __init__(self, input_size, hidden_size, drop_prob, basic_net='LSTM', batch_first=True, hidden=None):
        super(TemporalNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden = hidden
        if basic_net == 'LSTM':
            self.RNN = nn.LSTM(self.input_size,
                               self.hidden_size,
                               num_layers=2,  # 双层LSTM
                               dropout=drop_prob,
                               batch_first=batch_first,
                               bidirectional=False)  # 改为双向网络
        else:
            self.RNN = nn.RNN(self.input_size,
                              self.hidden_size,
                              num_layers=1,
                              dropout=drop_prob,
                              batch_first=batch_first,
                              bidirectional=False)
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        # Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了，防止过拟合

    def forward(self, input_tensor, sequence_lens):
        # 2023-4-18 程序并行化改造加入参数 enforce_sorted=False
        pack_inputs = pack_padded_sequence(input_tensor, sequence_lens.cpu(),
                                           batch_first=True, enforce_sorted=False)
        # 将一个填充过的变长序列压紧。按列压，pack之后，原来填充的 PAD（一般初始化为0）占位符被删掉了
        # 这里添加.cpu()的原因：torch.nn.utils.pack_padded_sequence: remove hidden cross-device copy for lengths (#41984) In
        # previous versions, when the lengths argument was a CUDA tensor, it would incorrectly be moved to the CPU
        # silently. This can lead to surprising performances and CPU/GPU sync when using CUDA so this has been
        # removed. You need to make sure that the provided lengths is a CPU Tensor when it is provided as a Tensor.

        # 这里的self.RNN是一条完整的LSTM链，而不是单个的LSTM CELL，因此self.hidden是作为初始化的细胞状态输入的，缺省时细胞状态均为0
        # 需要手动更新细胞状态的是nn.LSTMCell
        if self.hidden is not None:
            out, hiddens = self.RNN(pack_inputs, self.hidden)
            print('hidden is not None')
        else:
            out, hiddens = self.RNN(pack_inputs)
            # print('hidden:',1)
        last_hiddens, _ = pad_packed_sequence(out,
                                              batch_first=True)  # 这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。填充时会初始化为0
        last_hiddens = self.dropout(last_hiddens)

        return last_hiddens, hiddens


class Spatio_Temporal_Module(nn.Module):
    def __init__(self, input_module_size, input_semanticmodule_size, input_historymodule_size,
                 hidden_size, is_attn, config, hidden=None):
        super(Spatio_Temporal_Module, self).__init__()
        self.hidden_size = hidden_size
        self.his_hidden_size = int(hidden_size * 0.5)
        self.is_attn = is_attn
        # build the net
        self.temporal_net = TemporalNet(input_size=input_module_size,  # 单LSTM,输入拼接
                                        hidden_size=hidden_size,
                                        drop_prob=config['dropout'],
                                        hidden=hidden)
        '''self.gat_temporal_net = TemporalNet(input_size=input_gatmodule_size,
                                            hidden_size=hidden_size,
                                            hidden=hidden)'''
        self.semantic_temporal_net = TemporalNet(input_size=input_semanticmodule_size,
                                                 hidden_size=hidden_size,
                                                 drop_prob=config['dropout'],
                                                 hidden=hidden)
        # 这里还不能随意更改
        '''
        self.history_temporal_net = TemporalNet(input_size=input_historymodule_size,
                                                hidden_size=int(hidden_size * 0.5),
                                                hidden=hidden)
        '''
        self.spatio_attn_net = AttentionNet.Spatial_AttentionNet(config=config)
        # xinzheng
        # self.history_attn_net = AttentionNet.History_AttentionNet()
        self.scorer = AttentionNet.Scorer()
        # xinzheng
        # self.history_scorer = AttentionNet.History_Scorer()

    def end_dim(self):
        return self.hidden_size

    def history_end_dim(self):
        return self.his_hidden_size

    def forward(self, input_tensor, traj, input_semtensor, semantic):
        # spa_attn_score = self.scorer(traj, history)  # 这里的spa_attn_score是由轨迹的速度，转向角，行驶距离等组成的特征矩阵
        spa_attn_score = self.scorer(traj)  # 这里的spa_attn_score是由轨迹的速度，转向角，行驶距离等组成的特征矩阵
        # xinzheng
        # his_attn_score = self.history_scorer(history, traj)
        sequence_lens = traj['lens']
        # print("sequence_lens:")
        # print(sequence_lens)
        # history_lens = traj['his_lens']
        # sorted_history_lens, indices = torch.sort(history_lens, descending=True)
        # print("history_lens:")
        # print(sorted_history_lens)

        # #单LSTM,输入拼接
        # LSTM_input=torch.cat((input_tensor,input_gattensor),dim=2)
        # sptm_hiddens, hiddens = self.temporal_net(LSTM_input, sequence_lens)
        # sptm_lstmhiddens=sptm_hiddens#这两个变量只是为了返回时不报错，后面不涉及计算
        # sptm_gathiddens=sptm_hiddens#这两个变量只是为了返回时不报错，后面不涉及计算

        sptm_lstmhiddens, hiddens = self.temporal_net(input_tensor, sequence_lens)
        # sptm_gathiddens, gathiddens = self.gat_temporal_net(input_gattensor, sequence_lens)
        sptm_semhiddens, semhiddens = self.semantic_temporal_net(input_semtensor, sequence_lens)
        # input_historytensor = input_historytensor.view(-1, 94, 32, 128)
        # 出现了大小不符合的问题
        # sptm_historyhiddens, historyhiddens = self.history_temporal_net(input_historytensor, sorted_history_lens)
        # sptm_hiddens = torch.mul(sptm_lstmhiddens, sptm_gathiddens)  # 将两个LSTM的结果相乘
        sptm_hiddens = torch.mul(sptm_lstmhiddens, sptm_semhiddens)  # 将两个LSTM的结果相乘
        '''
        print("sptm_hiddens.shape")
        print(sptm_hiddens.shape)
        print("\n")
        print("sptm_historyhiddens.shape")
        print(sptm_historyhiddens.shape)
        print("\n")
        # sptm_hiddens = torch.cat((sptm_hiddens, sptm_historyhiddens), dim = 1)
        # print('calling the concatenate module!')    误差反向传播，此函数会调用多次
        print(sptm_hiddens.shape)
        '''
        # sptm_hiddens = torch.cat((sptm_lstmhiddens, sptm_gathiddens),dim=1)
        if self.is_attn:
            # print("sptm_hiddens.shape:")
            # print(sptm_hiddens.shape)
            # print("spa_attn_score.shape:")
            # print(spa_attn_score.shape)
            sptm_out, weights = self.spatio_attn_net(sptm_hiddens, spa_attn_score)
            # history_out, weights = self.history_attn_net(sptm_historyhiddens, his_attn_score)
            # print("sptm_out.shape:")
            # print(sptm_out.shape)
            # print("history_out.shape:")
            # print(history_out.shape)
            # 连接后输入一个目的地预测模型
            # 历史出行活动学习——移至单独模块
            # sptm_out = torch.cat((sptm_out, history_out), dim=1)
            # 历史预测结果张量和当前预测结果张量求平均值，是否效果好于连接后输入一个模型
            # sptm_out = 0.5 * (sptm_out + history_out)
        else:
            # sptm_out = sptm_hiddens.sum(dim=1)  # (B x S x H), get the last output of hiddens
            sptm_out = sptm_hiddens[:, -1, :]
            weights = None
        # print(sptm_out.size(),sptm_gathiddens.size())
        return sptm_out, sptm_lstmhiddens, weights, sptm_semhiddens
