from torch.utils import data
import torch
import pandas as pd
import os
import numpy as np
from .preprocessing import * 
from pdb_clone import pdb
from settings import PROJECT_ROOT, DATA_DIR

def normalize(data):
    if data.dim() == 1:
        return (data - torch.mean(data))/ torch.std(data)
    else:
        mean = torch.mean(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        std = torch.std(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        return  (data - mean) / std


class weather_data(data.Dataset):
    def __init__(self, root=PROJECT_ROOT+'/data/tmp/'):
        # TODO: remove first 3 lines of weather forecast for each file
        #       weather forecast files are from 2020
        self.weather_dirs_ = [root + str(dir_) for dir_ in os.listdir(root)]
        self.data, self.time_frame = self.load_data(self.weather_dirs_)
        # self.lead_time = 
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
        return "Weather Data : " + str(self.data.shape)



class wind_data_v2(data.Dataset):
    '''
    difference between v1 v2 is that preprocessing happens inside the class
    '''
    def __init__(self, window=5, ltime=18, wind_dir=PROJECT_ROOT+"/data/wind_energy.csv"):
        '''
        Attributes:
            data : torch.Tensor
            time_frame = np.ndarray() // time is stored in str type
        '''
        self.wind_dir = wind_dir
        self.lead_time = ltime
        self.window = window

        self.target_day = 18
        self.data, self.time_frame, self.raw = self.load_data(wind_dir)
        self.data, self.time_frame, self.raw= self.to_difference()

    def to_difference(self):
        ltime= self.lead_time
        t_0 = self.data[ltime:]
        # past by lead time 
        t_h = self.data[:-ltime]

        raw_t_0 = self.raw[ltime:]
        # past by lead time 
        raw_t_h = self.raw[:-ltime]

        return t_0 - t_h , self.time_frame[ltime:], raw_t_0 - raw_t_h
    def load_data(self,dirs_):

        data_np = pd.read_csv(dirs_).values
        time = data_np[:,1]
        energy_np = data_np[:,2].astype(np.float64)
        energy = torch.Tensor(energy_np)
        raw = energy.clone()
        # normalize energy generated
        energy = normalize(energy)
        return energy, time, raw

    def collect_window(self,data,idx):
        window = self.window
        indices = range(window, idx.stop - idx.start, 1 if idx.step==None else idx.step)
        out = []
        for i in iter(indices):
            out.append(data[i - window : i].unsqueeze(0))
        return torch.cat(out,axis=0)
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.format(idx)

    def format(self,idx):
        '''
        Note that the starting point of the data is 2 * lead_time  (i.e. x(t=0) = row (2*leadtime))
        This is because of calculating force,momentum
        '''
        ltime= self.lead_time
        window = self.window
        # pdb.set_trace()
        start = 0 if idx.start == None else idx.start
        end = self.data.shape[0] if idx.stop == None else idx.stop
        # slice considering window
        idx = slice(2 * ltime+ start , end + 2 * ltime, idx.step)
        idx2 = slice(start, 2 * ltime + end, idx.step)
        idx3 = slice(start + 2 * ltime - window, end + 2 * ltime, idx.step)
        idx4 = slice(start + 2 * ltime + self.target_day, end + 2 * ltime + self.target_day, idx.step)

        m,f = difference_orders(self.data[idx2], ltime)
        time_features = extract_time_feature(self.time_frame[idx]) 
        window_data = self.collect_window(self.data[idx3], idx3)

        formatted_x = torch.cat([window_data, m, f, time_features], axis=1)
        y = self.data[idx4]

        assert formatted_x.shape[0] == y.shape[0]
        return formatted_x, y 

    def __repr__(self):
        return "Wind Data : " + str(self.data.shape)

class final_dataset(data.Dataset):
    def __init__(self, window=5, ltime=18):
            '''
            Attributes:
                data : torch.Tensor
                time_frame = np.ndarray() // time is stored in str type
            '''
            self.lead_time = ltime
            self.window = window

            self.target_day = 18
            
            # tmppath = os.path.join(PROJECT_ROOT + DATA_DIR + "/tmp")

            self.wind_data = wind_data_v2()
            self.weather_data = weather_data()
    def __getitem__(self,idx):
        return self.format(idx)

    def format(self, idx):
        wind_x, y = self.wind_data[idx]
        weather_x = self.weather_data[idx]
        x = self.concat(wind_x, weather_x)
        return x,y
        
    # TODO: To be done after weather dataset
    # def concat(self, wind_x, weather_x):

# class wind_data(data.Dataset):
#     def __init__(self, wind_dir="../data/wind_energy.csv"):
#         '''
#         Attributes:
#             data : torch.Tensor
#             time_frame = np.ndarray() // time is stored in str type
#         '''
#         self.wind_dir = wind_dir
#         self.data, self.time_frame = self.load_data(wind_dir)
        
#     def load_data(self,dirs_):

#         data_np = pd.read_csv(dirs_).values
#         time = data_np[:,1]
#         energy_np = data_np[:,2].astype(np.float64)
#         energy = torch.Tensor(energy_np)
#         # normalize energy generated
#         energy = normalize(energy)      
#         return energy, time

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self,idx):
#         return self.data[idx]

#     def __repr__(self):
#         return "Wind Data : " + str(self_)



# def load_dataset(window=5,ratio=0.2):
#     wind_dataset = wind_data()
#     wind_dataset = preprocess(wind_dataset)

#     weather_dataset = weather_data()
#     # TODO: Any weather preprocessing ? 
#     # weather_dataset = preprocess_weather_data(weather_dataset)

#     x, y = concat_dataset(wind_dataset, weather_dataset,window) 
#     return test_split(x,y,ratio)


if __name__ == "__main__":
    dataset = wind_data_v2()
    print(dataset[:10])
    print(dataset[:10][0].shape, dataset[:10][1].shape)
#     print(dataset.data[54:59],'\n\n_________________\n',dataset[0:5], dataset)