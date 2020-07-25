import pandas as pd
import datetime
import calendar
import numpy as np

BIN_SIZE = 6
BIAS_WEIGHT = 0.1

def clean_result_data(result_csv):
    df = pd.DataFrame(result_csv)
    df.columns = ['time', 'actual', 'forecast', 'sale', 'buy', 'penalty', 'wasted', 'cih']

    chiwah_index = df.index[df['time'] == 'Chiwah'].tolist()[0]

    start_index = chiwah_index+ 3

    current_index = start_index
    current_time = df.iloc[current_index]['time']

    while not current_time == None:
        current_index += 1 
        current_time = df.iloc[current_index]['time']
        
    last_index = current_index - 1

    our_past_scores_df = df.loc[start_index:last_index]
    our_past_scores_df['time'] = pd.to_datetime(our_past_scores_df['time'])

    return our_past_scores_df

def get_useful_times():
    """
        return current utc time, time that we have to perdict, and a day before the prediction time
    """
    time_now = datetime.datetime.utcnow()
    time_pred = time_now + datetime.timedelta(hours=18)
    time_pred_one_day_before = time_now - datetime.timedelta(days=1)

    return time_now, time_pred, time_pred_one_day_before

def change_date_to_str(day, month, year):
    """
        change day, month, year to a str format as in result_csv
    """
    return str(day) + '-' + calendar.month_abbr[month] + '-' + str(year) + ' ' + '00:00UTC'

def get_time_frame_for_bias_calc(our_result_df, start_index):
    """
        return time from in which the bias is calculated
    """
    time_frame_index_for_bias_calc = our_result_df.index[our_result_df['time'] == start_index].tolist()[0]
    time_frame_for_bias_calc = our_result_df.loc[time_frame_index_for_bias_calc:time_frame_index_for_bias_calc + 23]
    time_frame_for_bias_calc['diff'] = time_frame_for_bias_calc['actual'].astype(int) - time_frame_for_bias_calc['forecast'].astype(int)
    
    return time_frame_for_bias_calc, time_frame_index_for_bias_calc


def make_bias_bin(time_frame_for_bias_calc, time_frame_index_for_bias_calc):
    bin = {}

    for i in range(len(time_frame_for_bias_calc) // BIN_SIZE):
        bin_range = i * BIN_SIZE
        current_bin_start_index = time_frame_index_for_bias_calc + bin_range
        current_bin_mean = np.mean(time_frame_for_bias_calc['diff'].loc[current_bin_start_index:current_bin_start_index + BIN_SIZE - 1])
        
        bin[i] = current_bin_mean * BIAS_WEIGHT

    return bin

def calculate_bias(bias_bin, time_pred):
    key = 0
    if time_pred.hour >= 0 and time_pred.hour <= 5:
        key = 0
    elif time_pred.hour >= 6 and time_pred.hour <= 11:
        key = 1
    elif time_pred.hour >=12 and time_pred.hour <= 17:
        key = 2
    elif time_pred.hour >=18 and time_pred.hour <=23:
        key = 3

    return bias_bin[key]

def get_bias(result_csv):
    our_result_df = clean_result_data(result_csv)
    time_now, time_pred, time_pred_one_day_before = get_useful_times()

    # retrieve time frame from which the bias is calculated
    start_date_pred_one_day_before_str = change_date_to_str(time_pred_one_day_before.day, time_pred_one_day_before.month, time_pred_one_day_before.year)
    time_frame_for_bias_calc, time_frame_index_for_bias_calc = get_time_frame_for_bias_calc(our_result_df, start_date_pred_one_day_before_str)
    
    # get bias bin from which the bias is calculated
    bin = make_bias_bin(time_frame_for_bias_calc, time_frame_index_for_bias_calc)

    return calculate_bias(bin, time_pred)