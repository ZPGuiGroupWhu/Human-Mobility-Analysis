# Project Introduction

This project (in *Code/*) is the code of Neurocomputing 2021 paper ***LSI-LSTM: An Attention-aware LSTM for Real-time Driving Destination Prediction by Considering Location Semantics and Location Importance of Trajectory Points***. 

We provide the complete version of code and part of sample data used in the paper. You can easily replace it with your own data in the same format. See the samples in *Code/data/* for more details.

The directory structure of the project is as follows：

|-- LSI-LSTM
    |-- README.md
    |-- Code
    |   |-- hyper_params_config.py
    |   |-- main.py
    |   |-- __init__.py
    |   |-- data
    |   |   |-- Driver_1.txt
    |   |-- model
    |   |   |-- LSI_LSTM.py
    |   |   |-- __init__.py
    |   |   |-- modules
    |   |       |-- __init__.py
    |   |       |-- destination_prediction_module
    |   |       |   |-- destination_prediction.py
    |   |       |   |-- residual_net.py
    |   |       |   |-- __init__.py
    |   |       |-- input_module
    |   |       |   |-- Input.py
    |   |       |   |-- __init__.py
    |   |       |-- travel_pattern_module
    |   |           |-- spatial_attention.py
    |   |           |-- travel_pattern_learning.py
    |   |           |-- __init__.py
    |   |-- model_save
    |   |-- py_utils
    |       |-- get_dataloader.py
    |       |-- loss.py
    |       |-- to_var.py
    |       |-- train.py
    |       |-- train_test_load.py
    |       |-- __init__.py
    |-- Publications

- README.md: the introduction document
- Code/hyper_params_config.py: the hyperparameter setting of LSI-LSTM
- Code/main.py: the entry of the model Training & evaluation
- Code/data: the storage directory of sample data
- Code/model: the source code of LSI-LSTM
- Code/model/modules: the three modules of LSI-LSTM
- Code/mode_save: the directory to save the trained model
- Code/py_utils: some python scripts used to assist model training
- Publications: the source file of the paper

# Usage

## Model Training & Evaluation

python main.py

### Hyperparameter

The hyperparameter of this project is set in Code/*hyper_params_config.py*

```
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
```

- data_path: the path of sample data used in the project; you can replace it with your own data in the same format
- BATCH_SIZE: the batch size to train, default 128
- EVAL_BATCH_SIZE: the evaluation_batch_size to test, default 64
- device: training way, is on CUDA or CPU
- lr: the learning rate of model training
- epochs: the epoch to train, default 100
- hidden_size: the hidden size of the model, default 128
- clip_gradient: the clip threshold of gradient to prevent the gradient exploding problem, default 3
- model_save_path: the path to save the trained model parameters

## How to Use Your Own Data

In the data folder *Code/data* we provide a sample data of one driver, i.e., *Driver_1.txt*. You can use your own data with the corresponding format as in the data sample. The sampled data contains 2158 trajectories. To make the model performance close to our proposed result, make sure your dataset contains more than 5M trajectories.

### Format Instructions

The format of the sample data is a JSON string, one line of the data represents a complete travel trajectory record. The sub-trajectory segmentation code used to simulate the driver travel precess is in *Code/py_utils/train_test_load.py*. The snapshot of the sample is as follows:

```
{"dis_total": 4.033611, 
"sem_O": [0.227161, 0.191744, 0.046635, 0.0, 0.16827, 0.050494, 0.084907, 0.711287],
"destination": [0.250521, -0.019345],
"lngs": [-0.111007, -0.101381, ...],
"lats": [0.259581, 0.268703, ...],
"travel_dis": [0, 0.104746, ...], 
"spd": [1.656881, 5.281216, ...], 
"azimuth": [0, 1.808405, ...], 
"weekday": 0, 
"start_time": 35, 
"norm_dict": [114.250663, 0.086436, 22.744553, 0.051849],
"sem_pt": [5, 5, ...]}
```

- dis_total: total distance of the travel (km)
- sem_O: the location semantics of departure region, each dimension represents the importance of the corresponding POI type
- destination: the real destination of the trajectory
- lngs: the normalized longitude sequence of the trajectory points
- lats: the normalized latitude sequence of the trajectory points
- travel_dis: the traveled distance sequence of the trajectory (km)
- spd: the driving speed sequence of the trajectory (m/s)
- azimuth: the turning angle sequence of the trajectory, from 0 to Π
- weekday: the weekday index, from 0 to 6
- start_time: the departure time slot, from 0 to 47
- norm_dict: the dictionary to transform the normalized lngs and lats to the form of original record, \[lngs_medium, lngs_std, lats_medium, lats_std\]
- sem_pt: the location semantics sequence of trajectory points, using the most importance POI type, from 0 to 8, 0 means unknown

The GPS points in a trajectory should be resampled with nearly equal distance.

Furthermore, replace the config file according to your own data, including the lngs_medium, lngs_std, lats_medium, lats_std, etc.
