This directory contains the code for this method. The role of each file is listed below.
code
  |--HTS_LDAVEC_PRED_model.py    is the overall structure of our model;
  |--data_Loader.py              contains the steps of sub-trajectory generation and dataset partitioning;
  |--HTS_LDAVEC_PRED             contains four sripts, each of which is a module of our model.
    |--Input_Module.py           processes input features, including current traveling features and historical activity chain.
    |--Spatial_Temporal_Module.py  implements the current movement mode learning module.
    |--History_Module.py         implements the historical movement mode learning module.
    |--AttentionNet.py           conducts spatiotemporal scoring mechanism.
    |--Destination_Prediction_Module.py    outputs the coordinates of destination in the form of longitude and latitude.
