import torch
import torch.nn as nn
import torch.nn.functional as F


class Spatial_Attention_Layer(nn.Module):

    def __init__(self):
        super(Spatial_Attention_Layer, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU()
        )

    def forward(self, hidden, score):
        weights = F.softmax(self.projector(score), dim=1)
        atten_result = (hidden * weights).sum(dim=1)
        return atten_result, weights


class Scorer(nn.Module):
    def forward(self, traj):
        # the location importance is proportionate to the azimuth and travel_dis, and in inverse proportion to the speed
        spd = (1.0 / (traj['spd'] + 1.0)).unsqueeze(-1)
        azimuth = (traj['azimuth']).unsqueeze(-1)
        dis = traj['travel_dis'].unsqueeze(-1)
        score = torch.cat((spd, azimuth, dis), dim=2)
        return score