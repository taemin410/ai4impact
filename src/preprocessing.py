from pdb_clone import pdb
import torch

def test_split(x, y, ratio=0.2):
    split_idx = int(x.shape[0] * (1- ratio))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def extract_time_feature(time_frame):
    '''
    Input:
        time_frame = np.ndarray with shape (N,)
    Output:
        output = torch.Tensor with shape (N - 2 * ltime, 36)
                Month, hour are the two features extracted
    '''

    # extract month and hour
    features = torch.Tensor([ [time.month, time.hour]  for time in time_frame]).long()
    zero_time = torch.Tensor([0] * 24)
    zero_month = torch.Tensor([0] * 12)
    time_list = []
    month_list = [] 
    # convert to one hot encoding format
    for i in range(time_frame.shape[0]):
        one_hot_time = zero_time.clone()
        one_hot_month = zero_month.clone()
        
        month = features[i][0]
        time = features[i][1]

        one_hot_time[time-1] = 1
        one_hot_month[month-1] = 1

        time_list.append(one_hot_time)
        month_list.append(one_hot_month)
    time = torch.stack(time_list)
    month = torch.stack(month_list)
    return torch.cat([time,month], axis=1)


def difference_orders(series, ltime):
    '''
    Input:
        series = torch.Tensor, time series series with shape (N,)
        ltime = lead time (hrs)
    Output:
        momentum, force = first, second order differences
    '''
    x_t_2h = series[:-2*ltime]
    x_t_h = series[ltime:-ltime]
    x_t_0 = series[ltime *2 : ]
    return (x_t_0 - x_t_h).unsqueeze(1), (x_t_0 - 2 * x_t_h + x_t_2h).unsqueeze(1)

def preprocess(dataset, ltime = 18):
    '''
    Input:
        dataset = torch.utils.Dataset, time series wind dataset 
        ltime = lead time
    Output:
        preprocessed dataset
    '''
    m,f = difference_orders(dataset, ltime)
    # m = momentum(dataset.data, ltime).unsqueeze(1)
    # f = force(dataset.data, ltime).unsqueeze(1)
    time_features = extract_time_feature(dataset.time_frame, ltime) 
    assert time_features.shape[0] == f.shape[0]
    assert f.shape == m.shape 

    dataset.data = torch.cat([dataset.data[2*ltime:].unsqueeze(1), m, f, time_features], axis=1)

    return dataset

if __name__ == "__main__":
    pass