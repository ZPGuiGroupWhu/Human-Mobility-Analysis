import torch
import torch.nn as nn
from .residual_net import ResidualNet
import torch.nn.functional as F


class Destination_Prediction_Moudle(nn.Module):
    def __init__(self, input_size, drop_prob=0):
        super(Destination_Prediction_Moudle, self).__init__()

        self.residual_net = ResidualNet(input_size)

        self.input2hid = nn.Linear(128, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 2)

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, sptm_out, traj_semantic):
        out = self.residual_net(torch.cat((sptm_out, traj_semantic), dim=1))

        out = F.leaky_relu(self.input2hid(out))
        active_out = F.leaky_relu(self.hid2hid(out))

        result = self.hid2out(self.dropout(active_out))

        return result