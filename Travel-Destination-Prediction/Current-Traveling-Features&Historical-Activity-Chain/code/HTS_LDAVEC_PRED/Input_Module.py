#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 19:07
# @Author: zhangxiaotong
# @File  : Input_Module.py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rc('font', family='SimHei', size=13)  # family为想要的字体,size就是可视图的中文字体大小

# torch.backends.cudnn.enabled = False
# 训练模块中才需要用到，这里屏蔽
'''
# 用于设置随机初始化的种子，即上述的编号，编号固定，每次获取的随机数固定。
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# config = {}
# device_set = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_set = torch.device("cuda")
'''
'''
config['clip_gradient'] = 3
# config['save_every'] = 4
# config['input_size'] = 3
config['output_size'] = 2
config['lr'] = 1e-3
config['epochs'] = 90
config['BATCH_SIZE'] = 32
config['EVAL_BATCH_SIZE'] = 32
config['hidden_size'] = 128
# config['machine'] = 'google'
config['device'] = device_set
'''


class Input_LstmModule(nn.Module):
    """
    Embedding the start_time and weekday
    这里不需要修改，因为特征维度没有增减
    """

    def __init__(self):
        super(Input_LstmModule, self).__init__()  # 强继承

        # 词典的大小尺寸num_embeddings、嵌入向量的维度embedding_dim和填充id，需要注意的是，如果符号总共有500个，指定了padding_idx，那么num_embeddings应该为501
        # 当用nn.Embedding()进行词向量嵌入时，对应的索引为padding_idx的向量将变为全为0的向量。这样就减少了填充值对模型训练的影响。

        self.fc = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
            nn.Linear(8, 3, False),  # 全连接层，第三个参数为bias - 如果设置为False，则图层不会学习附加偏差。默认值：True
        )

        # 输入参数为Module.add_module(name: str, module: Module)。功能为，为Module添加一个子module，对应名字为name。

    def end_dim(self):
        end_dim = 12
        # accroding to the length of input_tensor
        return end_dim

    def sem_dim(self):
        return 3 + 6

    def forward(self, attr, traj):
        # print(traj)
        lngs = traj['lng'].unsqueeze(-1)
        lats = traj['lat'].unsqueeze(-1)
        locs = torch.cat((lngs, lats), dim=2)

        input_tensor = torch.cat((locs,
                                  traj['travel_dis'].unsqueeze(-1),
                                  traj['spd'].unsqueeze(-1),
                                  traj['v_mean'].unsqueeze(-1),
                                  traj['v_std'].unsqueeze(-1),
                                  traj['angle'].unsqueeze(-1),
                                  traj['time_consume'].unsqueeze(-1),
                                  # traj['time_stamp_s'].unsqueeze(-1),
                                  # traj['time_stamp_c'].unsqueeze(-1),
                                  traj['start_time_s'].unsqueeze(-1),
                                  traj['start_time_c'].unsqueeze(-1),
                                  traj['weekday_s'].unsqueeze(-1),
                                  traj['weekday_c'].unsqueeze(-1)), dim=2)

        return input_tensor, traj


class Input_GatModule(nn.Module):
    """这个模块可以取消了"""
    def __init__(self):
        super(Input_GatModule, self).__init__()
        self.fc = nn.Sequential(nn.Linear(8, 3, False), )
        # nn.ReLU()

    def end_dim(self):
        end_dim = 14
        return end_dim

    def forward(self, gat):
        lngs = gat['lng'].unsqueeze(-1)
        lats = gat['lat'].unsqueeze(-1)
        locs = torch.cat((lngs, lats), dim=2)

        input_tensor = torch.cat((locs,
                                  gat['d1_lng'].unsqueeze(-1),
                                  gat['d1_lat'].unsqueeze(-1),
                                  gat['d1_a2'].unsqueeze(-1),
                                  gat['d2_lng'].unsqueeze(-1),
                                  gat['d2_lat'].unsqueeze(-1),
                                  gat['d2_a2'].unsqueeze(-1),
                                  gat['d3_lng'].unsqueeze(-1),
                                  gat['d3_lat'].unsqueeze(-1),
                                  gat['d3_a2'].unsqueeze(-1),
                                  gat['d4_lng'].unsqueeze(-1),
                                  gat['d4_lat'].unsqueeze(-1),
                                  gat['d4_a2'].unsqueeze(-1)), dim=2)

        return input_tensor, gat  # (B x end_dim)


class Input_SemanticModule(nn.Module):
    """
    Embedding the start_semantic and current_semantic
    语义输入模块需要修改
    """

    def __init__(self):
        super(Input_SemanticModule, self).__init__()
        # 由于语义输入维度变化，维度由8改成5
        # self.fc = nn.Sequential(nn.Linear(5, 3, False), )
        # 词向量已经提前降维，这里的FC取消
        # self.wordvec_fc = nn.Sequential(nn.Linear(256, 5, False), )
        # nn.ReLU()

    def end_dim(self):
        # 这里增加了词向量的维度64，所以维度由10改为74
        end_dim = 5
        # end_dim = 74        # 改变，不进行Embedding，直接输入74维向量
        # end_dim = 8
        # accroding to the length of input_tensor
        return end_dim

    def forward(self, semantic):
        lngs = semantic['lng'].unsqueeze(-1)
        lats = semantic['lat'].unsqueeze(-1)
        locs = torch.cat((lngs, lats), dim=2)
        '''
        loc_semantic = torch.cat((semantic['semantic_0'].unsqueeze(-1),
                                  semantic['semantic_1'].unsqueeze(-1),
                                  semantic['semantic_2'].unsqueeze(-1),
                                  semantic['semantic_3'].unsqueeze(-1),
                                  semantic['semantic_4'].unsqueeze(-1)), dim=2)
                                  '''
        # semantic['semantic_5'].unsqueeze(-1),
        # semantic['semantic_6'].unsqueeze(-1),
        # semantic['semantic_7'].unsqueeze(-1)), dim=2)
        # loc_semantic_tensor = self.fc(loc_semantic)
        # 得到一个dim-3向量
        # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
        # 是因为维度没有增加1的问题
        word_vec_tensor = semantic['word_vec_0'].unsqueeze(-1)
        # 词向量====3维
        # 语义主题===5维
        # 经纬度===2维
        for i in range(1, 3):
            word_vec_tensor = torch.cat((word_vec_tensor, semantic['word_vec_' + str(i) + ''].unsqueeze(-1)), dim=2)
        # word_vec_decline_tensor = self.wordvec_fc(word_vec_tensor)
        # 这里就不用降维啦
        input_tensor = torch.cat((locs, word_vec_tensor), dim=2)
        '''
        input_tensor = torch.cat((locs,
                                  semantic['semantic_0'].unsqueeze(-1),
                                  semantic['semantic_1'].unsqueeze(-1),
                                  semantic['semantic_2'].unsqueeze(-1),
                                  semantic['semantic_3'].unsqueeze(-1),
                                  semantic['semantic_4'].unsqueeze(-1),
                                  semantic['semantic_5'].unsqueeze(-1),
                                  semantic['semantic_6'].unsqueeze(-1),
                                  semantic['semantic_7'].unsqueeze(-1),
                                  ), dim=2)
        # 在这里添加词向量的键和值
        for i in range(64):
            input_tensor = torch.cat((input_tensor, semantic['word_vec_' + str(i) + ''].unsqueeze(-1)), dim=2)
        '''
        return input_tensor, semantic  # (B x end_dim)


class Input_HistoryModule(nn.Module):
    """
    Embedding the start_semantic and current_semantic
    """

    def __init__(self):
        super(Input_HistoryModule, self).__init__()
        self.fc = nn.Sequential(nn.Linear(8, 3, False), )
        # nn.ReLU()

    def end_dim(self):
        # 由于增加了时间语义，输入Tensor维度由10增加为14
        # 由于语义主题维度变成5，输入Tensor维度由14减少为11
        end_dim = 11
        # accroding to the length of input_tensor
        return end_dim

    def forward(self, history):
        lngs = history['history_lng'].unsqueeze(-1)
        lats = history['history_lat'].unsqueeze(-1)
        locs = torch.cat((lngs, lats), dim=2)

        input_tensor = torch.cat((locs,
                                  history['history_semantic_0'].unsqueeze(-1),
                                  history['history_semantic_1'].unsqueeze(-1),
                                  history['history_semantic_2'].unsqueeze(-1),
                                  history['history_semantic_3'].unsqueeze(-1),
                                  history['history_semantic_4'].unsqueeze(-1),
                                  # history['history_semantic_5'].unsqueeze(-1),
                                  # history['history_semantic_6'].unsqueeze(-1),
                                  # history['history_semantic_7'].unsqueeze(-1),
                                  history['history_weekday_sin'].unsqueeze(-1),
                                  history['history_weekday_cos'].unsqueeze(-1),
                                  history['history_start_time_sin'].unsqueeze(-1),
                                  history['history_start_time_cos'].unsqueeze(-1)), dim=2)

        return input_tensor, history  # (B x end_dim)
