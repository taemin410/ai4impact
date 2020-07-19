import pytest
import os
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
from settings import PROJECT_ROOT, DATA_DIR

import pandas as pd
from src import dataset
# import pdb
import torch
import numpy as np

def test_data(test, expected_output):
    idx = test[1]
    dataset = test[0]
    if isinstance(idx,int):
        loaded = dataset[idx]
        print(loaded[0].dtype, expected_output[0])
        print("Dataset's value\n\n",loaded[0] ,'\n\nGround Truth\n\n',expected_output[0])
        print(loaded[0] == expected_output[0] , loaded[1] == expected_output[1])
        false_idx = (1 - loaded[0] == expected_output[0].int()).nonzero()
        # assert dataset[idx][0] == expected_output[0] and dataset[idx][1] == expected_output[1] 
    else:
        print("Dataset's value\n\n",dataset[idx[0] : idx[1]][0] ,'\n\nGround Truth\n\n',expected_output[0])
        print(dataset[idx[0] : idx[1]][0] == expected_output[0], dataset[idx[0] : idx[1]][1] == expected_output[1])
        # assert dataset[idx[0] : idx[1]][0] == expected_output[0] and dataset[idx[0] : idx[1]][1] == expected_output[1] 

ltime = 10
window = 5
# Note that one must change the global variable values of src/dataset.py
test1 = [dataset.final_dataset(window=window,ltime=ltime,root=PROJECT_ROOT+'/test_data/', difference=0,normalize=0), 0]
test2 = [dataset.final_dataset(window=window,ltime=ltime,root=PROJECT_ROOT+'/test_data/', difference=0,normalize=0), [10,12]]

# test3 = [dataset.final_dataset(root='../test_data/', difference=1), 0]
# test4 = [dataset.final_dataset(root='../test_data/', difference=1), [10,12]]
df = pd.read_csv('../test_data/wind_energy_v2.csv')
# pdb.set_trace()
window1 = df['energy'][ltime * 2 - window + test1[1]: ltime * 2 + test1[1]].values
window1 = np.expand_dims(window1,axis=0)

window2 = []
for i in range(test2[1][0],test2[1][1]):
    add = df['energy'][ltime * 2 - window + i: ltime * 2 + i].tolist()
    window2.append(add)    
window2 = np.array(window2)
assert window2.shape[0] == test2[1][1] - test2[1][0]

one_hot_month = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
one_hot_time_20 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
one_hot_time_6 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
one_hot_time_7 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

df1 = pd.read_csv('../test_data/history_cleaned/angerville-1.csv')
# df2 = pd.read_csv('../test_data/history_cleaned/angerville-1-b.csv')

forecast1 = df1[['Speed(m/s)',"Direction (deg N)"]].iloc[list(range(3,11))].values
forecast1[:,1] = forecast1[:,1] * np.pi / 180
forecast1 = torch.Tensor(forecast1)
# cos1 = np.expand_dims(np.cos(forecast1[:,1]),axis=1)
# sin1 = np.expand_dims(np.sin(forecast1[:,1]),axis=1)
cos1 = np.expand_dims(torch.cos(forecast1[:,1]),axis=1)
sin1 = np.expand_dims(torch.sin(forecast1[:,1]),axis=1)

forecast1 = np.expand_dims(np.concatenate([np.expand_dims(forecast1[:,0],axis=1), cos1, sin1],axis=1).reshape(-1), axis=0)
assert forecast1.shape[1] == 24

forecast2 = []
for i in range(test2[1][0],test2[1][1]):
    i = (i + ltime *2)//6
    row = df1[['Speed(m/s)',"Direction (deg N)"]].iloc[list(range(i,i+8))].values
    angle = row[:,1] / 180 * np.pi
    speed = np.expand_dims(row[:,0],axis=1)
    cos = np.expand_dims(np.cos(angle), axis=1)
    sin = np.expand_dims(np.sin(angle), axis=1)
    forecast2.append(np.concatenate([speed, cos, sin], axis=1).reshape(-1))    
forecast2 = np.array(forecast2)

output1 = [
    torch.Tensor(np.concatenate([
        window1, np.array([[0,0]]), \
        one_hot_time_20 , one_hot_month, np.array([[0,0]]), \
        forecast1
        ],axis=1)
    ),
    torch.Tensor([500])
]
window2_2 = torch.tensor([0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  5.0000e+02])
avg = torch.mean(window2_2)
std = torch.std(window2_2)
features2 = np.concatenate([window2, \
            np.array([[500, 500],[0,0]]), \
            np.concatenate([one_hot_time_6, one_hot_time_7], axis=0), np.concatenate([one_hot_month, one_hot_month], axis=0),\
            np.array([[0,0],[avg,std]]),forecast2
        ], axis=1)

output2 = [
    torch.Tensor(features2),
    torch.Tensor([3500, 3500])
]

print(test_data(test1,output1))
print(test_data(test2,output2))

# @pytest.mark.parametrize("Data_loading_scenario,expected_output", [
#     (test1, output1), \
#     (test2, output2)  \
# ])

