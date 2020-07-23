from torch.utils import data
import torch
import pandas as pd
import datetime
from datetime import datetime as dt
import numpy as np
import sys, os

from .preprocessing import *
if "/src" in sys.path[0]:
    from preprocessing import *
# else:
#     from src.preprocessing import *
from pdb_clone import pdb

sys.path.insert(0, os.path.abspath(os.path.join("..")))

from settings import PROJECT_ROOT, DATA_DIR
from torch.utils.data import SequentialSampler


def normalize(data):
    if data.dim() == 1:
        return (data - torch.mean(data))/ torch.std(data), torch.mean(data), torch.std(data)
    else:
        mean = torch.mean(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        std = torch.std(data, axis=1).unsqueeze(1).repeat(1,data.shape[1])
        return  (data - mean) / std

class weather_data(data.Dataset):
    def __init__(
        self,
        root=PROJECT_ROOT + DATA_DIR + "/history_cleaned/",
        version=0,
        time_interval=6,
        normalize=1,
    ):
        self.weather_dirs_ = [root + str(dir_) for dir_ in os.listdir(root)]
        self.normalize = normalize
        self.x_mean = []
        self.x_std = []

        self.data, self.time_frame = self.load_data(self.weather_dirs_, version)
        self.time_interval = time_interval
        self.version = version
        self.last_idx = min([region.shape[0] for region in self.data])
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx - 1, idx, 1)

        return [region[idx] for region in self.data]

    def load_data(self, dirs_, version=0):
        """
        Input:
            dirs_
            version - 0= load both forecast models
                      1 = load forecast model 1
                      2 = load forecast model 2
        """
        forcast = []
        times = []
        # pdb.set_trace()
        if version == 1:
            dirs_ = [dir_ for dir_ in dirs_ if "-b" not in dir_]
        elif version == 2:
            dirs_ = [dir_ for dir_ in dirs_ if "-b" in dir_]
        for dir_ in dirs_:
            tmp = pd.read_csv(dir_, header=0)
            # tmp["Time"] = tmp["Time"].apply(
            #     lambda x: dt.strptime(x[2:-3] + ":00", "%y-%m-%d %H:%M:%S")
            # )
            tmp = tmp.values
            time = tmp[:, 1]

            speed_direction_np = tmp[:, 2:].astype(np.float64)
            speed_direction = torch.Tensor(speed_direction_np)
            # normalize speed
            if self.normalize:
                speed_direction[:, 0], mean, std = normalize(speed_direction[:, 0])
                self.x_mean.append(mean.item())
                self.x_std.append(std.item())
            speed_direction = torch.cat(
                [
                    speed_direction[:, 0].unsqueeze(1),
                    change_representation(speed_direction[:, 1]),
                ],
                axis=1,
            )
            assert speed_direction.shape[1] == 3

            forcast.append(speed_direction)
            times.append(time)
        return forcast, times

    def collect_forcast(self, idx, future=8, time_interval=6):
        """
        idx = index of __getitem__
        t_0 = x(T+0) frame 
        time_interval = time inverval of forcast model
        future = number of future frames to collect
        """
        out = []
        # pdb.set_trace()
        for i in range(idx.start, idx.stop):
            forecast = [
                region[i // time_interval : i // time_interval + future].reshape(-1)
                for region in self.data
            ]
            add = torch.cat(forecast, axis=0)
            out.append(add)
        out = torch.stack(out)
        # number of region * number of frames  *number of columns (direction, speed)
        # assert out.shape[1] == 8* (2 if self.version == 0 else 1)* future  *  3
        return out

    def __repr__(self):
        return "Weather Data : " + str(self.data.shape)

    def __len__(self):
        return self.last_idx


class wind_data_v2(data.Dataset):
    """
    difference between v1 v2 is that preprocessing happens inside the class
    """

    def __init__(
        self,
        last_idx,
        window=5,
        ltime=18,
        difference=1,
        wind_dir=PROJECT_ROOT + DATA_DIR + "/wind_energy.csv",
        normalize=1,
    ):
        """
        Attributes:
            data : torch.Tensor
            time_frame = np.ndarray() // time is stored in str type
        """
        self.wind_dir = wind_dir
        self.lead_time = ltime
        self.window = window
        self.normalize = normalize

        self.data, self.time_frame, self.raw = self.load_data(wind_dir)
        # if difference == 1:
        #     self.data, self.time_frame, self.raw = self.to_difference()
        self.first_idx = 2  * ltime
        self.last_idx = last_idx

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1, 1)
        return self.format(idx)

    def format(self, idx):
        """
        Note that the starting point of the data is 2 * lead_time  (i.e. x(t=0) = row (2*leadtime))
        This is because of calculating force,momentum
        """
        # pdb.set_trace()
        ltime = self.lead_time
        window = self.window

        start = idx.start
        end = idx.stop

        idx = slice(start, end, idx.step)
        idx2 = slice(start - 2 * ltime, end, idx.step)
        idx3 = slice(start - window + 1, end, idx.step)
        idx4 = slice(start + ltime, end + ltime, idx.step)

        time_features = extract_time_feature(self.time_frame[idx])
        m, f = difference_orders(self.data[idx2], ltime)
        window_data = self.collect_window(self.data[idx3], idx3)
        window_avg, window_std = self.window_stats(window_data)
        y = self.data[idx4]
        formatted_x = torch.cat(
            [window_data, m, f, time_features, window_avg, window_std], axis=1
        )

        assert formatted_x.shape[0] == y.shape[0]
        # window column + difference orders +  time features(month, time) of frame T+0
        # assert formatted_x.shape[1] == (self.window + 2 + 36 + 2)

        return formatted_x, y

    def window_stats(self, data):
        return (
            torch.mean(data, axis=1).unsqueeze(1),
            torch.std(data, axis=1).unsqueeze(1),
        )

    def load_data(self, dirs_):

        data = pd.read_csv(dirs_)
        data["time"] = data["time"].apply(
            lambda x: dt.strptime(x[2:], "%y-%m-%d %H:%M:%S")
        )
        data_np = data.values
        time = data_np[:, 1]
        energy_np = data_np[:, 2].astype(np.float64)
        energy = torch.Tensor(energy_np)
        raw = energy.clone()
        # normalize energy generated
        if self.normalize == 1:
            energy, self.x_mean, self.x_std = normalize(energy)
        return energy, time, raw

    def to_difference(self):
        ltime = self.lead_time
        t_0 = self.data[ltime:]
        # past by lead time
        t_h = self.data[:-ltime]

        raw_t_0 = self.raw[ltime:]
        # past by lead time
        raw_t_h = self.raw[:-ltime]

        return t_0 - t_h, self.time_frame[ltime:], raw_t_0 - raw_t_h

    def collect_window(self, data, idx):
        # pdb.set_trace()

        window = self.window
        indices = range(
            window, idx.stop - idx.start + 1, 1 if idx.step == None else idx.step
        )
        out = []
        # pdb.set_trace()
        for i in iter(indices):
            out.append(data[i - window : i].unsqueeze(0))
        return torch.cat(out, axis=0)
 
    def __repr__(self):
        return "Wind Data : " + str(self.data.shape)

    def __len__(self):
        return self.last_idx - self.first_idx


class final_dataset(data.Dataset):
    def __init__(
        self, window=5, ltime=18, difference=1, version=0, root=None, normalize=1):
        """
            Attributes:
                data : torch.Tensor
                time_frame = np.ndarray() // time is stored in str type
                window
                ltime = lead time
                differnce = convert target & energy production input to difference values 
                version = which forecast models to include should be one of (0,1,2)
            """
        self.lead_time = ltime
        self.window = window

        self.difference = difference
        self.normalize = normalize
        if root != None:
            self.weather_data = weather_data(
                version=version, root=root + "/history_cleaned/", normalize=0
            )
            last_idx = self.weather_data.last_idx * 6  - 48
            self.wind_data = wind_data_v2(
                window=window,
                ltime=ltime,
                difference=difference,
                wind_dir=root + "/wind_energy_v2.csv",
                normalize=0,
            )
        else:
            self.weather_data = weather_data(
                version=version,
                root=PROJECT_ROOT + DATA_DIR + "/history_cleaned/",
                normalize=normalize,
            )
            last_idx = self.weather_data.last_idx * 6  - 48
            self.wind_data = wind_data_v2(
                last_idx,
                window=window,
                ltime=ltime,
                difference=difference,
                wind_dir=PROJECT_ROOT + DATA_DIR + "/wind_energy_v2.csv",
                normalize=normalize,
            )
        self.first_idx = 2  * ltime
        # maximum index that has a target
        self.last_idx = last_idx

    def collect_weather(self, idx):
        return self.weather_data.collect_forcast(idx)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1, 1)
        if isinstance(idx, list):
            idx = slice(min(idx), max(idx) + 1, 1)

        return self.format(idx)

    def __len__(self):
        return self.last_idx - self.first_idx

    def format(self, idx):
        # pdb.set_trace()
        start = 0 if idx.start == None else idx.start
        end = self.last_idx if idx.stop == None else idx.stop
        if start >= 0:
            add = 1
        # apply warmup time
        start += self.last_idx if start < 0 else self.first_idx
        end += self.last_idx if end < 0 else self.first_idx

        # when timeframe has no target
        if end > self.last_idx:
            end = self.last_idx
        idx = slice(start, end, idx.step)
        # print(idx)
        # print("Timeframe considered x(T+0) :", self.wind_data.time_frame[idx])
        wind_x, y = self.wind_data[idx]

        weather_x = self.collect_weather(idx)
        # print(wind_x.shape, weather_x.shape)
        x = torch.cat([wind_x, weather_x], axis=1)

        return x, y


def load_dataset(
    window=10,
    ltime=18,
    difference=1,
    version=0,
    split_ratio=0.2,
    val_ratio=0.2,
    batch_size=16,
    root=None,
    normalize=1,
):
    """
    Input:
        window
        ltime
        difference = (0 or 1) whether to make use of difference target
        version = (0,1,2) which forecast model to make use of 0 = both version 
                  1 = version1 only
                  2 = version2 only
        split_ratio = train/test split ratio
        val_ratio = train/val split ratio (train/val is split from the training dataset constructed by the ratio of split_ratio)
        batch_size 
    """
    if root is not None:
        dataset = final_dataset(window, ltime, difference, version, root, 0)
    else:
        dataset = final_dataset(window, ltime, difference, version, None, normalize)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_ratio) * dataset_size))
    val_split = int(split * (1 - val_ratio))

    split = -1
    # split idxs
    train_indices, test_indices = indices[:split], indices[split:]
    train_indices, val_indices = train_indices[:val_split], train_indices[val_split:]
    # create sampler
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    train_dataset = torch.utils.data.TensorDataset(
        train_dataset[0].clone().detach(), train_dataset[1].clone().detach()
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_dataset[0].clone().detach(), val_dataset[1].clone().detach()
    )
    test_dataset = torch.utils.data.TensorDataset(
        test_dataset[0].clone().detach(), test_dataset[1].clone().detach()
    )
    # Creating data samplers
    train_sampler = SequentialSampler(train_dataset)
    valid_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)
    # Create Loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler
    )

    return train_loader, validation_loader, test_loader, dataset.wind_data.x_mean, dataset.wind_data.x_std, dataset.weather_data.x_mean, dataset.weather_data.x_std 

def load_latest(window=5, ltime=18 ,x_mean=0, x_std=1, forecast_mean=[0]*16, forecast_std=[1]*16):
    wind_data = pd.read_csv(PROJECT_ROOT+ DATA_DIR+'/wind_energy_v2.csv', header=0)
    wind_data['energy'] = (wind_data['energy'] - x_mean) / x_std
    wind_data["time"] = wind_data["time"].apply(
        lambda x: dt.strptime(x[2:], "%y-%m-%d %H:%M:%S")
    )
    # window
    window_data = wind_data['energy'].iloc[-window:].tolist()
    window_data = torch.Tensor(window_data) 
    window_avg = torch.mean(window_data).unsqueeze(0)
    window_std = torch.std(window_data).unsqueeze(0)
    
    # time feature
    last_row = wind_data.iloc[-1]
    zero_time = torch.Tensor([0] * 24)
    zero_month = torch.Tensor([0] * 12)
    
    zero_month[last_row['time'].month -1] = 1
    zero_time[last_row['time'].hour -1] = 1
    time_feature = torch.cat([zero_time, zero_month], axis=0)  
    # difference features
    x_t_h = torch.Tensor([wind_data['energy'].iloc[-ltime]])
    x_t_2h = torch.Tensor([wind_data['energy'].iloc[-2 * ltime]])
    x_t_0 = torch.Tensor([wind_data['energy'].iloc[-1]])
 
    momentum = x_t_0 - x_t_h
    force = x_t_0 - 2 * x_t_h + x_t_2h

    dirs_ = os.listdir(PROJECT_ROOT + DATA_DIR + "/forecast")
    forecast_data = [pd.read_csv(PROJECT_ROOT + DATA_DIR + "/forecast/"+ dir_) for dir_ in dirs_ ]
    forecast_features = []

    print('Retrieving features of timeframe : ',last_row['time'])
    for idx_, (region, mean, std) in enumerate(zip(forecast_data, forecast_mean, forecast_std)):
        region['Time'] = region['Time'].apply(lambda x: dt.strptime(x[2:-3] + ":00", "%y-%m-%d %H:%M:%S"))
        region['Speed(m/s)'] = (region['Speed(m/s)'] - mean)/std
                
        region_data = []
        imputation_n_retreive = 0
        for i, row in region.iterrows():
            time_diff = (row['Time'] - last_row['time']) /np.timedelta64(1,'h')
            if  time_diff < 9 and time_diff > 0:
                if region.shape[0] - i > 16:    
                    for j in range(8):
                        region_data.append(region[['Speed(m/s)','Direction (deg N)']].iloc[i+ 2*j].tolist())
                    forecast_features.append(region_data)
                else:
                    imputation_n_retreive = 1
                break
            elif i == region.shape[0] -1 and imputation_n_retreive == 0:
                imputation_n_retreive = 1
                 
        if imputation_n_retreive:
            added = 0
            for i, row in region[:-1].iterrows():
                row = pd.DataFrame(row).transpose()
                time_diff = int((region['Time'][i+1] - region['Time'][i])/np.timedelta64(1,'D') / 0.25 -1) 
                if time_diff > 0:
                    new_row = []
                    for j in range(time_diff):
                        row['Time'] = row['Time'].apply( lambda x : x + datetime.timedelta(hours=6))
                        new_row.append(row.copy())
                    region = pd.concat([region[:i+added+1]] + new_row + [region[i+added+1:]])
                    added += time_diff
            if region.shape[0] > 8:     
                forecast_features.append(region[['Speed(m/s)','Direction (deg N)']].iloc[-8:].values.tolist())
            else:
                # when there is less than 8 rows after imputation
                short_by = 8 - region.shape[0]

                to_add = region[['Speed(m/s)','Direction (deg N)']].values.tolist()
                for _ in range(short_by):
                    to_add.append(region[['Speed(m/s)','Direction (deg N)']].iloc[-1].tolist())
                forecast_features.append(to_add)

    # pdb.set_trace()
    forecast_features = torch.Tensor(forecast_features)
    forecast_features = forecast_features.reshape(-1,2)
    sin_cos = change_representation(forecast_features[:,1])
    forecast_features = torch.cat([forecast_features[:,0].unsqueeze(1), sin_cos], axis=1)
    forecast_features = forecast_features.reshape(-1)
    # when window = 5 
    # 5 + 1 + 1 + 36 + 1 + 1 + 16*8*3 = 429
    print([a.shape for a in [window_data, momentum, force, time_feature, window_avg, window_std, forecast_features]])
    return torch.cat([window_data, momentum, force, time_feature, window_avg, window_std, forecast_features], axis=0).unsqueeze(0)

if __name__ == "__main__":
    out = load_latest()
    print(out.shape)