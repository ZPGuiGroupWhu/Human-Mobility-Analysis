#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 21:19
# @Author: zhangxiaotong
# @File  : HTS_LDAVEC_model.py
import torch
import torch.nn as nn

from Model_new.HTS_LDAVEC_PRED import Destination_Prediction_Module
from Model_new.HTS_LDAVEC_PRED import Input_Module
from Model_new.HTS_LDAVEC_PRED import Spatial_Temporal_Module
from Model_new.HTS_LDAVEC_PRED import History_Module


class DeepSTC(nn.Module):

    def __init__(self, hidden_size, is_attn=True, hidden=None, config=None):
        super(DeepSTC, self).__init__()
        self.input_lstmmodule = Input_Module.Input_LstmModule()
        # self.input_gatmodule = Input_Module.Input_GatModule()
        # 增加地理语义表征
        self.input_semmodule = Input_Module.Input_SemanticModule()
        # 增加历史出行活动模块
        self.input_historymodule = Input_Module.Input_HistoryModule()
        self.spa_temp_module = Spatial_Temporal_Module.Spatio_Temporal_Module(
            input_module_size=self.input_lstmmodule.end_dim(),
            # input_gatmodule_size=self.input_gatmodule.end_dim(),
            input_semanticmodule_size=self.input_semmodule.end_dim(),
            input_historymodule_size=self.input_historymodule.end_dim(),
            hidden_size=hidden_size,
            hidden=hidden,
            config=config,
            is_attn=is_attn)
        self.config = config
        self.history_travel_module = History_Module.History_Module(config=config,
                                                                   input_size=self.input_historymodule.end_dim())
        self.dest_pre_module = Destination_Prediction_Module.Destination_Prediction_Moudle(
            input_size=self.spa_temp_module.end_dim() + self.history_travel_module.end_dim())
        self.dis_loss = Destination_Prediction_Module.DisLoss(is_MAE=True)
        self.sem_loss = Destination_Prediction_Module.SemLoss()

    def forward(self, attr, traj, semantic, history):
        input_tensor, traj = self.input_lstmmodule(attr, traj)
        # input_gattensor, gat = self.input_gatmodule(gat)
        input_semtensor, semantic = self.input_semmodule(semantic)
        input_historytensor, history = self.input_historymodule(history)
        # print(input_historytensor.shape)
        '''
        sptm_out, hiddens, weights, gathiddens, sptm_semhiddens, sptm_historyhiddens = self.spa_temp_module(
            input_tensor, traj, input_gattensor, gat, input_semtensor, semantic, input_historytensor, history)
        '''
        sptm_out, hiddens, weights, sptm_semhiddens = self.spa_temp_module(
            input_tensor, traj, input_semtensor, semantic)
        # 历史出行活动建模模块
        # query为长期活动语义链，key为近期活动语义链，value还要看具体的模型结构
        hist_out1, hist_out2, hist_out3, self_attn = self.history_travel_module(
            query=input_historytensor[:, -self.config['cp_lens']:, :],
            key=input_historytensor[:, :-self.config['cp_lens'], :],
            traj=traj, history=history)
        # sptm_hist_out = torch.cat((sptm_out, hist_out1, hist_out2, hist_out3), dim=1)
        sptm_hist_out = torch.cat((hist_out2, sptm_out), dim=1)
        result_lnglat, result_semantic = self.dest_pre_module(sptm_hist_out)
        # print(result)      Tensors
        # print('DeepSTC!')    此处也是会调用多次 不要在forward方法上试图输出
        return result_lnglat, result_semantic, hiddens, weights, self_attn

    # 这里相当于调用了forward函数，一种灵活的变体，等同于数据对象加括号直接调用
    # deep_tdp(attr, traj, gat, semantic)

    def eval_on_batch(self, attr, traj, semantic, history):
        # 这里的out就是二维的经纬度向量，现在可能要改成7维的经纬度+语义向量
        out_lnglat, out_semantic, hiddens, weights, self_attn = self(attr, traj, semantic, history)
        # torch.Size([32, 2])
        dest = attr['destination']

        # torch.Size([32, 5])
        truth_semantic = semantic['dest_semantic']
        # 增加判断类簇编号的代码
        cluster_id = attr['cluster_id']
        traj_id = attr['id']
        cut_length = attr['cut_length']
        dis_total = attr['dis_total']
        # cluster_num = None
        # 结构不需要大改
        # [[mean,std],[],[],...,[]]
        lngs_mean, lngs_std = attr['normlization_list'][0, 0], attr['normlization_list'][0, 1]
        lats_mean, lats_std = attr['normlization_list'][0, 2], attr['normlization_list'][0, 3]
        lngs_mean_xy, lngs_std_xy = attr['normlization_list_xy'][0, 0], attr['normlization_list_xy'][0, 1]
        lats_mean_xy, lats_std_xy = attr['normlization_list_xy'][0, 2], attr['normlization_list_xy'][0, 3]

        cur_pt = attr['cur_pt']
        # torch.Size([32, 2])
        # cur_pt = torch.cat((cur_pt, cur_pt), dim=0)
        # print("cur_pt.shape:")
        # print(cur_pt.shape)
        # print("out.shape:")
        # print(out.shape)

        # no destination semantic
        # Loss = self.dis_loss(cur_pt + out_lnglat, dest).mean()
        # This place [out_semantic, truth_semantic] is reversed, now is corrected!
        Loss = self.dis_loss(cur_pt + out_lnglat, dest).mean() + self.sem_loss(out_semantic, truth_semantic).mean()
        dest = torch.cat((dest[:, :1] * lngs_std_xy + lngs_mean_xy, dest[:, 1:2] * lats_std_xy + lats_mean_xy), dim=1)
        prd_d = torch.cat(((out_lnglat[:, :1] + cur_pt[:, :1]) * lngs_std + lngs_mean,
                           (out_lnglat[:, 1:2] + cur_pt[:, 1:2]) * lats_std + lats_mean), dim=1)
        # 增加目的地输出
        # print(dest) ——Tensor
        accuracy = 0

        # 定义为列表list
        '''
        delta_dis_cluster = []
        for cluster_id in range(cluster_num): ## None
            delta_dis_cluster.append([])
        '''
        # 这个一维Tensor定义不变，但是要分id存储结果
        delta_dis_total = Destination_Prediction_Module.get_dis(prd_d, dest)
        RSE = torch.pow(delta_dis_total, 2)
        RE = delta_dis_total / attr['dis_total']
        # 先把Tensor转为list
        total_dis_list = delta_dis_total.detach().cpu().numpy().tolist()
        cluster_id_list = cluster_id.detach().cpu().numpy().tolist()
        traj_id_list = traj_id.detach().cpu().numpy().tolist()
        cut_length_list = cut_length.detach().cpu().numpy().tolist()
        RSE_list = RSE.detach().cpu().numpy().tolist()
        RE_list = RE.detach().cpu().numpy().tolist()
        dis_total_list = dis_total.detach().cpu().numpy().tolist()
        # print(delta_dis_total)
        # 获取语义的预测结果
        if out_semantic.shape[1] == 5 and out_semantic.shape[0] == 32:
            pred = out_semantic.contiguous().view(-1, 5)
        else:
            pred = torch.zeros([32, 5], dtype=torch.float).view(-1, 5).cuda()
        if truth_semantic.shape[1] == 5 and truth_semantic.shape[0] == 32:
            truth = truth_semantic.contiguous().view(-1, 5)
        else:
            truth = torch.zeros([32, 5], dtype=torch.float).view(-1, 5).cuda()
        semantic_dis_total = Destination_Prediction_Module.get_similarity(pred, truth)
        semantic_list = semantic_dis_total.detach().cpu().numpy().tolist()
        each_traj_result = []
        for i in range(len(total_dis_list)):
            result_dict = {"id": traj_id_list[i],
                           "cluster_id": cluster_id_list[i],
                           "cut_length": cut_length_list[i],
                           "dis_total": dis_total_list[i],
                           "AE": total_dis_list[i],
                           "RE": RE_list[i],
                           "RSE": RSE_list[i],
                           "semE": semantic_list[i]
                           }
            each_traj_result.append(result_dict)
        '''
        for i in range(len(total_dis_list)):
            # 判断destination属于哪个类簇，查字典
            # 假设k是字典查到的结果，即第k个簇
            k = None
            delta_dis_cluster[k].append(total_dis_list[i])
        '''

        # 以下这些需要重定义，之前是一维列表，现在要改成二维数组
        # 之前是一个数值，现在是一维列表，每个值代表着某一类簇的精度
        '''
        MAE_Loss_rep = []
        RMSE_Loss_rep = []
        MRE_Loss_rep = []
        for i in range(cluster_num):
            # 把列表中的每一个list转为Tensor
            delta_dis_Tensor_cluster = torch.tensor(delta_dis_cluster[i])
            # 用Tensor进行计算
            MAE_Loss_rep.append(delta_dis_Tensor_cluster.mean())
            RMSE_Loss_rep.append(torch.pow(delta_dis_Tensor_cluster, 2).mean())
            MRE_Loss_rep.append((delta_dis_Tensor_cluster / attr['dis_total']).mean())
        '''
        MAE_Loss_rep = delta_dis_total.mean()
        RMSE_Loss_rep = torch.pow(delta_dis_total, 2).mean()
        # MRE_Loss_rep = (delta_dis / (attr['dis_total'] - traj['travel_dis'][:, -1])).mean()
        #  delta_dis m  dis_total km
        MRE_Loss_rep = (delta_dis_total / attr['dis_total']).mean()
        Semantic_Loss_rep = semantic_dis_total.mean()
        # 返回值增加一个所有轨迹预测结果
        return Loss, MRE_Loss_rep, MAE_Loss_rep, RMSE_Loss_rep, Semantic_Loss_rep, accuracy, each_traj_result, self_attn
        # 这里的accuracy未进行操作，恒为0
