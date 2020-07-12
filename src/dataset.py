from torch.utils import data
import torch
import pandas as pd
from datetime import datetime as dt

import numpy as np
# from .preprocessing import *
from preprocessing import *

from pdb_clone import pdb
import sys,os
sys.path.insert(0,os.path.abspath(os.path.join('..')))
from settings import PROJECT_ROOT, DATA_DIR

def normalize(data):
    if data.dim() == 1:
        return (data - torch.mean(data))/ torch.std(data)
    else:
        mean = torch.mean(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        std = torch.std(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        return  (data - mean) / std


class weather_data(data.Dataset):
    def __init__(self, root=PROJECT_ROOT + DATA_DIR + "/history_cleaned/",version=0,time_interval=6):
        self.weather_dirs_ = [root + str(dir_) for dir_ in os.listdir(root)]
        self.data, self.time_frame = self.load_data(self.weather_dirs_,version)
        self.time_interval = time_interval
        self.version = version
        # self.lead_time = 


    def __getitem__(self,idx):
        if isinstance(idx,int):
            idx = slice(idx-1 , idx, 1)

        return [region[idx] for region in self.data]

    def load_data(self,dirs_,version=0):
        '''
        Input:
            dirs_
            version - 0= load both forecast models, 1 = load model 1, 2 = load model 2
        '''
        forcast = []
        times = []
        if version == 1:
            dirs_ = [dir_ for dir_ in dirs_ if '-b' not in dir_]
        elif version == 2:
            dirs_ = [dir_ for dir_ in dirs_ if '-b' in dir_]

        for dir_ in dirs_:
            tmp = pd.read_csv(dir_)
            tmp['Time'] = tmp['Time'].apply(lambda x : dt.strptime(x[2:-3]+":00", '%y/%m/%d %H:%M:%S'))
            tmp = tmp.values
            time = tmp[:,0]

            speed_direction_np = tmp[:,1:].astype(np.float64)
            speed_direction = torch.Tensor(speed_direction_np)
            # normalize speed 
            speed_direction[:,0] = normalize(speed_direction[:,0])
            forcast.append(speed_direction)      
            times.append(time)
            
        return forcast, times

    def collect_forcast(self, idx, future=8, time_interval=6):
        '''
        idx = index of __getitem__
        time_interval = time inverval of forcast model
        future = number of future frames to collect
        '''
        out = []
        # pdb.set_trace()
        for i in range(idx.start, idx.stop):
            forecast = [ region[i//time_interval: i//time_interval + future].reshape(-1) for region in self.data]
            add = torch.cat(forecast,axis=0)
            print(i,i//time_interval,add.shape)
            out.append(add)

        out = torch.stack(out)
        # number of region * number of frames  *number of columns (direction, speed) 
        print(out.shape[1], 8* (2 if self.version == 0 else 1)* future  *  2 )
        assert out.shape[1] == 8* (2 if self.version == 0 else 1)* future  *  2 
        return out

    def __repr__(self):
        return "Weather Data : " + str(self.data.shape)
        
    def __len__(self):
        return self.data[0].shape[0]




class wind_data_v2(data.Dataset):
    '''
    difference between v1 v2 is that preprocessing happens inside the class
    '''
    def __init__(self, window=5, ltime=18, difference=1, wind_dir=PROJECT_ROOT + DATA_DIR + "/wind_energy.csv"):
        '''
        Attributes:
            data : torch.Tensor
            time_frame = np.ndarray() // time is stored in str type
        '''
        self.wind_dir = wind_dir
        self.lead_time = ltime
        self.window = window
        self.data, self.time_frame, self.raw = self.load_data(wind_dir)
        if difference == 1: 
            self.data, self.time_frame, self.raw= self.to_difference()
        self.last_idx = 5114 * 6 - 48
        
    def __getitem__(self,idx):
        if isinstance(idx,int):
            idx = slice(idx-1 , idx, 1)

        return self.format(idx)

    def format(self,idx):
        '''
        Note that the starting point of the data is 2 * lead_time  (i.e. x(t=0) = row (2*leadtime))
        This is because of calculating force,momentum
        '''
        # pdb.set_trace()
        ltime= self.lead_time
        window = self.window
        start = 0 if idx.start == None else idx.start
        end = self.last_idx if idx.stop == None else idx.stop
        if start < 0:
            start += self.last_idx
        if end < 0:
            end += self.last_idx
        # when timeframe has no target
        if end > self.last_idx :
            end =  self.last_idx 
        

        idx = slice(2 * ltime+ start , end + 2 * ltime, idx.step)
        idx2 = slice(start, 2 * ltime + end, idx.step)
        idx3 = slice(start + 2 * ltime - window, end + 2 * ltime, idx.step)
        idx4 = slice(start + 3 * ltime , end + 3 * ltime, idx.step)

        window_data = self.collect_window(self.data[idx3], idx3)
        m,f = difference_orders(self.data[idx2], ltime)
        time_features = extract_time_feature(self.time_frame[idx]) 
        
        formatted_x = torch.cat([window_data, m, f, time_features], axis=1)
        y = self.data[idx4]
        assert formatted_x.shape[0] == y.shape[0]
        # window column + difference orders +  time features(month, time) of frame T+0 
        assert formatted_x.shape[1] == (5 + 2 + 36)

        return formatted_x, y 

    def load_data(self,dirs_):

        data = pd.read_csv(dirs_)
        data['time'] = data['time'].apply(lambda x : dt.strptime(x[2:], '%y-%m-%d %H:%M:%S')) 
        data_np = data.values       
        time = data_np[:,1]
        energy_np = data_np[:,2].astype(np.float64)
        energy = torch.Tensor(energy_np)
        raw = energy.clone()
        # normalize energy generated
        energy = normalize(energy)
        return energy, time, raw

    def to_difference(self):
        ltime= self.lead_time
        t_0 = self.data[ltime:]
        # past by lead time 
        t_h = self.data[:-ltime]

        raw_t_0 = self.raw[ltime:]
        # past by lead time 
        raw_t_h = self.raw[:-ltime]

        return t_0 - t_h , self.time_frame[ltime:], raw_t_0 - raw_t_h


    def collect_window(self,data,idx):
        window = self.window
        indices = range(window, idx.stop - idx.start, 1 if idx.step==None else idx.step)
        out = []
        # pdb.set_trace()
        for i in iter(indices):
            out.append(data[i - window : i].unsqueeze(0))
        return torch.cat(out,axis=0)

    def __repr__(self):
        return "Wind Data : " + str(self.data.shape)

            
    def __len__(self):
        return self.data.shape[0]



class final_dataset(data.Dataset):
    def __init__(self, window=5, ltime=18, difference=1, version=0):
            '''
            Attributes:
                data : torch.Tensor
                time_frame = np.ndarray() // time is stored in str type
                window
                ltime = lead time
                differnce = convert target & energy production input to difference values 
                version = which forecast models to include should be one of (0,1,2)
            '''
            self.lead_time = ltime
            self.window = window
            # tmppath = os.path.join(PROJECT_ROOT + DATA_DIR + "/tmp")
            self.difference = difference
            self.wind_data = wind_data_v2(window=window,ltime=ltime,difference=difference)
            self.weather_data = weather_data(version=version)
            # maximum index that has a target
            self.last_idx = 5114 * 6 - 48#min([a.data.shape[0] * 6 for a in self.weather_data.data ]) - 8

    def collect_weather(self, idx):
        return self.weather_data.collect_forcast(idx)

    def __getitem__(self,idx):
        if isinstance(idx,int):
            idx = slice(idx-1 , idx, 1)
        return self.format(idx)

    def format(self, idx):
        # pdb.set_trace()
        wind_x, y = self.wind_data[idx]
        
        add = 0
        start = 0 if idx.start == None else idx.start
        end = self.last_idx if idx.stop == None else idx.stop
        if start > 0 and end > 0:
            add = 1
        elif start < 0:
            start += self.last_idx  
        if end < 0:
            end += self.last_idx  
        # when timeframe has no target
        if end > self.last_idx :
            end =  self.last_idx 
        print("Timeframe considered x(T+0) :", self.wind_data.time_frame[self.lead_time * 2 + start : self.lead_time * 2 + end])

        # (2 + self.difference) because adding differnce removes first ltime rows (i.e. shift (T+0) timeframe by ltime)
        warmup = (2 + self.difference) * self.lead_time
        idx = slice( warmup * add + start, warmup * add + end, idx.step)
        weather_x = self.collect_weather(idx)
        x = torch.cat([wind_x, weather_x], axis=1)

        return x,y
        

if __name__ == "__main__":
    dataset = final_dataset(difference=1,version=0)
    x , y= dataset[-3:]
    # print(x.shape, y.shape)
    x , y= dataset[1:3]
    # print(x.shape, y.shape)
    
    # print(dataset[:1])

    # error in this one is due to the date range difference in forcast and energy history
    # print(dataset[-100:-90])
    # window + month_time(one hot) + 16 region * 8 future forecast * 2 (wind,direction)
    # 5 + 36  2+ 16 * 8 * 2 = 235