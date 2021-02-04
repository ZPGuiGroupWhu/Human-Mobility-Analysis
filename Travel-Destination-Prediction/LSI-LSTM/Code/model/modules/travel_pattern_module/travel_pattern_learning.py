import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .spatial_attention import Spatial_Attention_Layer, Scorer


class Travel_Pattern_Learning_Module(nn.Module):

    def __init__(self, input_module_size, hidden_size, hidden=None):
        super(Travel_Pattern_Learning_Module, self).__init__()
        self.hidden_size = hidden_size
        self.temporal_net = TemporalNet(input_size=input_module_size,
                                        hidden_size=hidden_size,
                                        hidden=hidden)
        self.spatio_attn_layer = Spatial_Attention_Layer()
        self.scorer = Scorer()

    def end_dim(self):
        return self.hidden_size

    def forward(self, input_tensor, traj):
        spa_attn_score = self.scorer(traj)
        sequence_lens = traj['lens']
        tp_hiddens, hiddens = self.temporal_net(input_tensor, sequence_lens)
        sptm_out, weights = self.spatio_attn_layer(tp_hiddens, spa_attn_score)

        return sptm_out, tp_hiddens, weights


class TemporalNet(nn.Module):

    def __init__(self, input_size, hidden_size=128, drop_prob=0, batch_first=True, hidden=None):
        super(TemporalNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden = hidden
        self.RNN = nn.LSTM(self.input_size,
                           self.hidden_size,
                           num_layers=2,
                           dropout=drop_prob,
                           batch_first=batch_first,
                           bidirectional=False)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input_tensor, sequence_lens):
        pack_inputs = pack_padded_sequence(input_tensor, sequence_lens, batch_first=True)
        if self.hidden is not None:
            out, hiddens = self.RNN(pack_inputs, self.hidden)
        else:
            out, hiddens = self.RNN(pack_inputs)
        last_hiddens, _ = pad_packed_sequence(out, batch_first=True)
        last_hiddens = self.dropout(last_hiddens)

        return last_hiddens, hiddens



