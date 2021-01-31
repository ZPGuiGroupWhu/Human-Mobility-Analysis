import torch

device_set = torch.device("cuda")
config = {}
config['data_path'] = './data/Driver_1.txt'
config['BATCH_SIZE'] = 128
config['EVAL_BATCH_SIZE'] = 64
config['device'] = device_set
config['lr'] = 1e-3
config['epochs'] = 100
config['hidden_size'] = 128
config['clip_gradient'] = 3
config['model_save_path'] = './model_save/'