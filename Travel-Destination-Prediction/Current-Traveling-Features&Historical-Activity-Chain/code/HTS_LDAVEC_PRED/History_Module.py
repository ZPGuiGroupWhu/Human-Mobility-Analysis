#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/2/23 18:59
# @Author: zhangxiaotong
# @File  : History_Module.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_new.HTS_LDAVEC_PRED import AttentionNet


class History_Module(nn.Module):
    def __init__(self, config, input_size):
        super(History_Module, self).__init__()
        self.config = config
        self.input_dim = input_size
        self.hidden_size = config['hist_hidden_size']
        self.history_attn_net = AttentionNet.History_AttentionNet(config=config)
        self.history_scorer = AttentionNet.History_Scorer(config=config)
        self.recently_scorer = AttentionNet.Recently_Score(config=config)
        self.temporal_module = His_Temporal_Module(input_size=input_size,
                                                   drop_prob=config['dropout'],
                                                   hidden_size=config['hist_hidden_size'])
        self.tf_encoder = DP_Transformer(config=config, input_dim=config['hist_hidden_size'])

    def end_dim(self):
        return self.hidden_size

    def forward(self, query, key, traj, history):
        # query - decoder; key/value - encoder
        history_lens = traj['his_lens']
        sorted_history_lens, _ = torch.sort(history_lens, descending=True)
        # print(key.size())
        # 以下两个参数对应 last_hiddens, hiddens
        sptm_history_hiddens, hiddens = self.temporal_module(input_tensor=key, hidden=None)
        his_attn_score = self.history_scorer(history, traj)
        out3, weights = self.history_attn_net(sptm_history_hiddens, his_attn_score)
        # query为长期活动语义链，key为近期活动语义链，value是经过LSTM学习得到的历史出行活动链
        out1, self_attn = self.tf_encoder(Q=query, K=key, V=sptm_history_hiddens)
        # attn over there is 注意力权重

        sptm_recent_hiddens, _ = self.temporal_module(input_tensor=query, hidden=hiddens)  #
        # 这里必须有两个参数承接返回值
        # 获得注意力得分的值
        recent_attn_score = self.recently_scorer(history, traj)
        out2, weights = self.history_attn_net(sptm_recent_hiddens, recent_attn_score)
        # out2 = out2[:, -1, :]
        return out1, out2, out3, self_attn


class His_Temporal_Module(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob=None, batch_first=True, bidirectional=True):
        super(His_Temporal_Module, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size if not bidirectional else int(hidden_size / 2)
        self.LSTM = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=2,
                            dropout=drop_prob,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input_tensor, hidden=None):
        if hidden is not None:
            last_hiddens, hiddens = self.LSTM(input_tensor, hidden)
        else:
            last_hiddens, hiddens = self.LSTM(input_tensor)
        last_hiddens = self.dropout(last_hiddens)

        return last_hiddens, hiddens


class DP_Transformer(nn.Module):

    def __init__(self, config, input_dim):
        super(DP_Transformer, self).__init__()
        self.input_dim = input_dim
        self.device = config['device']
        self.hidden_size = config['hist_hidden_size']
        self.ff_dim = config['ff_dim']
        self.layer = EncoderLayer(
            ff_dim=self.ff_dim,
            input_dim=self.input_dim,
        )

    def forward(self, Q, K, V):
        attn_self_mask = get_attn_pad_mask(Q, K).to(self.device)
        enc_output, enc_attn_self = self.layer(Q, K, V, attn_self_mask)
        # enc_output [batch_size, src_len, input_dim] ->  [batch_size, hidden_size]
        out = F.softmax(enc_output.sum(dim=1), dim=1)
        return out, enc_attn_self


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q, input_dim = seq_q.size()
    batch_size, len_k, input_dim = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data[:, :, 0].eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size()[-1])
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=1)(scores)
        # [batch_size, len_q, V_dim]
        context = torch.matmul(attn, V)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, ff_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim, bias=False)
        )
        self.layerNorm = nn.LayerNorm(self.input_dim)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, input_dim]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layerNorm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(EncoderLayer, self).__init__()
        self.attn = ScaledDotProductAttention()
        self.pos_ffn = PoswiseFeedForwardNet(input_dim=input_dim, ff_dim=ff_dim)

    def forward(self, Q, K, V, attn_self_mask):
        # enc_inputs: [batch_size, src_len, input_dim]
        enc_outputs, attn = self.attn(Q, K, V, attn_self_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
