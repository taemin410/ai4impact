from dataset import *
from pdb_clone import pdb

def test_split(x, y, ratio=0.2):
    '''
    Input:
        X
        Y
        ratio
    Output:
        train_x, train_y, test_x, test_y
    '''
    split_idx = int(x.shape[0] * (1- ratio))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def extract_time_feature(time_frame):
    '''
    Input:
        time_frame = np.ndarray with shape (N,)
    Output:
        output = torch.Tensor with shape (N - 2 * ltime, 2)
                Month, hour are the two features extracted
    '''

    time_frame = time_frame
    # extract month and hour
    features = torch.Tensor([ [int(time[5:7]), int(time[11:13])]  for time in time_frame]).long()
    zero_time = torch.Tensor([0] * 24)
    zero_month = torch.Tensor([0] * 12)
    time_list = []
    month_list = [] 
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


def momentum(series, ltime):
    '''
    Input:
        series = torch.Tensor, time series series with shape (N,)
        ltime = lead time (hrs)
    Output:
        momentum = torch.Tensor with shape (N-2 * ltime,)
                   (N - 2* ltime) for consistency with force
    '''
    t_past = series[2*ltime:]
    return series[ltime:-ltime] - t_past

def force(series, ltime):
    '''
    Input:
        series = torch.Tensor, time series series with shape (N,)
        ltime = lead time (hrs)
    Output:
        force = torch.Tensor(N- 2 * ltime,)
    '''
    t_past = series[ltime:-ltime]
    t_past_2 = series[ltime *2 : ]

    return series[:-2*ltime] - 2* t_past + t_past_2

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

# def preprocess_weather(dataset):
#     '''

#     '''
    

def concat_dataset(wind, weather, window=5, plus=18, forcast_future=6):
    '''
    Input:
        wind = torch.util.Dataset 
        weather = torch.util.Dataset
        window = window of past data
        plus = target y would be x(T+plus)
        forcast_future = number future weather forecast to be included
    Output:
        x = total_x 
        y = total_y
    '''
    # pdb.set_trace()
    y = wind[plus:,0]
    nrows = len(wind)
    x = torch.Tensor([])
    # for one hot encoding
    zero_time = torch.Tensor([[0] * 24])
    zero_month = torch.Tensor([[0] * 12])

    for i in range(window + plus, nrows):
        wind_past = wind[i-window- plus: i-plus, 0].unsqueeze(0)
        # force and momentum
        wind_features = wind[i, 1:3].unsqueeze(0)
        
        month = wind[i, -2].int()
        time = wind[i, -1].int()
        # convert to one hot encoding
        time_one_hot = zero_time; time_one_hot[0][time-1] = 1
        month_one_hot = zero_month; month_one_hot[0][month-1] = 1
        
        # TODO: include forecast data to node
        # weather_data = weather.data[]
        row = torch.cat([wind_past, wind_features, time_one_hot, month_one_hot],axis=1) # originally also weather_data

        if x.shape[0] == 0:
            x = row
        else:
            x = torch.cat([x,row], axis=0)

    return x, y



if __name__ == "__main__":
    trainx,trainy,testx,testy = load_dataset()
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)