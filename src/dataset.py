from torch.utils import data
import torch
import pandas as pd
import os
import numpy as np
def normalize(data):
    if data.dim() == 1:
        return (data - torch.mean(data))/ torch.std(data)
    else:
        mean = torch.mean(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        std = torch.std(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        return  (data - mean) / std

class wind_data(data.Dataset):
    def __init__(self, wind_dir="../data/wind_energy.csv"):
        '''
        Attributes:
            data : torch.Tensor
            time_frame = np.ndarray() // time is stored in str type
        '''
        self.wind_dir = wind_dir
        self.data, self.time_frame = self.load_data(wind_dir)
        
    def load_data(self,dirs_):

        data_np = pd.read_csv(dirs_).values
        time = data_np[:,1]
        energy_np = data_np[:,2].astype(np.float64)
        energy = torch.Tensor(energy_np)
        # normalize energy generated
        energy = normalize(energy)      
        return energy, time

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.data[idx]

    def __repr__(self):
        return "Wind Data : " + str(self_)


class weather_data(data.Dataset):
    def __init__(self, root='../data/tmp/'):
        # TODO: remove first 3 lines of weather forecast for each file
        #       weather forecast files are from 2020
        self.weather_dirs_ = [root + str(dir_) for dir_ in os.listdir(root)]
        self.data, self.time_frame = self.load_data(self.weather_dirs_)
        
    def load_data(self,dirs_):
        forcast = []
        times = []
        for dir_ in dirs_:
            tmp = pd.read_csv(dir_).values
            # normalize speed
            time = tmp[:,0]
            speed_direction_np = tmp[:,1:].astype(np.float64)
            speed_direction = torch.Tensor(speed_direction_np)
            speed_direction = normalize(speed_direction)
            
            forcast.append(speed_direction)      
            times.append(time)
            
        return forcast, times

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self,idx):
        return [region[idx] for region in self.data]

    def __repr__(self):
        return "Weather Data : " + str(self_)
