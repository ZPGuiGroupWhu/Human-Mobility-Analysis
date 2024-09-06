#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/2/23 19:04
# @Author: zhangxiaotong
# @File  : config_parameter.py
import torch

config = {}
# device_set = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_set = torch.device("cuda:0")
config['clip_gradient'] = 50
# config['save_every'] = 4
# config['input_size'] = 3
config['output_size'] = 2
config['lr'] = 1e-3
config['epochs'] = 80
config['num_processes'] = 5
config['BATCH_SIZE'] = 32
config['EVAL_BATCH_SIZE'] = 32
config['hidden_size'] = 128
config['hist_hidden_size'] = 32
config['machine'] = 'google'
config['device'] = device_set
config['dropout'] = 0.2
config['pt_nums'] = 200
config['cp_lens'] = 20
config['ff_dim'] = 64
