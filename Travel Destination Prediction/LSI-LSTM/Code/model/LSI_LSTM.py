import torch
import torch.nn as nn
from model.modules import Input_Module, Travel_Pattern_Learning_Module, Destination_Prediction_Moudle
from py_utils import get_dis, DisLoss


class LSI_LSTM(nn.Module):

    def __init__(self, hidden_size, hidden=None):
        super(LSI_LSTM, self).__init__()
        self.input_module = Input_Module()
        self.travel_pattern_module = Travel_Pattern_Learning_Module(input_module_size=self.input_module.end_dim(),
                                                                    hidden_size=hidden_size, hidden=hidden)
        self.dest_pre_module = Destination_Prediction_Moudle(input_size=self.travel_pattern_module.end_dim()
                                                                        + self.input_module.sem_dim())
        self.loss = DisLoss(is_MAE=True)

    def forward(self, attr, traj):
        input_tensor, traj, traj_semantic = self.input_module(attr, traj)
        sptm_out, hiddens, weights = self.travel_pattern_module(input_tensor, traj)
        result = self.dest_pre_module(sptm_out, traj_semantic)
        return result, hiddens, weights

    def eval_on_batch(self, attr, traj):
        out, hiddens, weights = self(attr, traj)
        dest = attr['destination']
        lngs_mean, lngs_std = attr['norm_dict'][0, 0], attr['norm_dict'][0, 1]
        lats_mean, lats_std = attr['norm_dict'][0, 2], attr['norm_dict'][0, 3]
        cur_pt = attr['cur_pt']

        Loss = self.loss(cur_pt + out, dest).mean()

        dest = torch.cat((dest[:, :1] * lngs_std + lngs_mean,
                          dest[:, 1:2] * lats_std + lats_mean), dim=1)

        prd_d = torch.cat(((out[:, :1] + cur_pt[:, :1]) * lngs_std + lngs_mean,
                           (out[:, 1:2] + cur_pt[:, 1:2]) * lats_std + lats_mean), dim=1)
        accuracy = 0

        delta_dis = get_dis(prd_d, dest)
        MAE_Loss_rep = delta_dis.mean()
        RMSE_Loss_rep = torch.pow(delta_dis, 2).mean()
        MRE_Loss_rep = (delta_dis / attr['dis_total']/1000).mean()

        return Loss, MRE_Loss_rep, MAE_Loss_rep, RMSE_Loss_rep, accuracy


