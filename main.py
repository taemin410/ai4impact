from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import final_dataset, load_dataset, load_latest
from src.utils.config import load_config
from src.model import NN_Model, Persistance
from src.trade.trader import Trader
from src.utils.logger import Logger
from src.eval import get_lagged_correlation
from src.utils.data import *
from script.download_data import download_data, parse_data

from src.request.request import submit_answer, get_result_from_web
import argparse
import torch
import threading
from datetime import datetime as dt
import datetime
import time
import os
import pandas as pd
import numpy as np 
from settings import PROJECT_ROOT, DATA_DIR
from prev import prev_val

RESUBMISSION_TIME_INTERVAL = 3600

def write_configs(writer, configs):
    configstr = ""
    for i in configs:
        configstr += str(i) + " : " + str(configs[i]) + "\n"
    writer.add_text("CONFIGS", configstr, 0)

def forecast_imputation():
    dirs_ = os.listdir( PROJECT_ROOT+ DATA_DIR+ '/history_cleaned')
    dirs_ = [PROJECT_ROOT+ DATA_DIR+'/history_cleaned/'+str(dir_) for dir_ in dirs_]
    for dir_ in dirs_:
        tmp = pd.read_csv(dir_)
        tmp["Time"] = tmp["Time"].apply(
                lambda x: dt.strptime(x[2:-3] + ":00", "%y-%m-%d %H:%M:%S")
        )
        added = 0
        # print(dir_)
        for i, row in tmp[:-1].iterrows():
                row = pd.DataFrame(row).transpose()
                time_diff = int((tmp['Time'][i+1] - tmp['Time'][i])/np.timedelta64(1,'D') / 0.25) -1 
                if time_diff != 0:
                        # print('\t',tmp['Time'][i])
                        future_speed = tmp['Speed(m/s)'][i+1]
                        current_speed = tmp['Speed(m/s)'][i]
                        future_direction = tmp['Direction (deg N)'][i+1]
                        current_direction = tmp['Direction (deg N)'][i]
                        
                        speed_step = (future_speed - current_speed)/ (time_diff + 1)
                        direction_step = (future_direction - current_direction) / (time_diff + 1)

                        new_row = []
                        for j in range(time_diff):
                            row['Time'] = row['Time'].apply( lambda x : x + datetime.timedelta(hours=6))
                            # row['Speed(m/s)'] += speed_step
                            # row['Direction (deg N)'] += direction_step
                            new_row.append(row.copy())
                        tmp = pd.concat([tmp[:i+added+1]] + new_row + [tmp[i+added+1:]])
                        added += time_diff                        
        tmp.to_csv(dir_,index=False)


def main(args):
    print(args)
    paths = download_data()
    for i in paths:
        parse_data(i)
    print("=============== Parsing dataset complete ===============")

    forecast_imputation()

    # Load configurations
    configs = load_config("config.yml")
    modelConfig = configs["model"]

    time = dt.now().strftime("%d-%m-%Y %H:%M:%S")
    logdir = "runs/" + time
    
    # # Initialize SummaryWriter for tensorboard
    writer = Logger(logdir)
    write_configs(writer, modelConfig)
  
    # # Preprocess the data
    train_loader, validation_loader, test_loader, data_mean, data_std, forecast_mean, forecast_std = load_dataset(
        difference=0,
        batch_size=modelConfig["batchsize"]
    )
    
    # # Baseline model
    # baseline_model = Persistance(18, writer)
    # initialize Model
    model = NN_Model(
        input_dim=train_loader.dataset.tensors[0].size(1),
        output_dim=1,
        hidden_layers=modelConfig["hiddenlayers"],
        writer=writer,
        device=args.device
    )

    model.train(
        train_loader,
        validation_loader,
        epochs=modelConfig["epochs"],
        lr=modelConfig["lr"],
        step_size=modelConfig["step_size"],
        gamma=modelConfig["gamma"],
        weight_decay=modelConfig["weight_decay"]
    )
    

    try:
        x = load_latest(10,18,data_mean.item(),data_std.item(), forecast_mean, forecast_std)

        ypred = model.predict(x)
        ypred  = (ypred * data_std.item()) + data_mean.item()
        print("Model running successful!")

    except Exception as err:
        print("Error message: ", err)
        ypred = args.prev
        print("model running failed... sending prev value")

    args.prev = ypred
    

    # b_rmse, b_ypred, b_ytest = baseline_model.test(test_loader)
    # rmse, ypred, ytest = model.test(test_loader)
    
    # print("RMSE:  ", rmse)
    # print("BASELINE: ", b_rmse) 

    # writer.add_text("RMSE", str(rmse.item()), 0)
    # writer.add_text("RMSE/Baseline", str(b_rmse.item()), 0)

    ####################
    # Lagged Corr      #
    ####################
    # lagged_vals = get_lagged_correlation(ypred = ypred, 
    #                                 ytrue = test_loader.dataset.tensors[1], 
    #                                 num_delta= 180 )
    # writer.draw_lagged_correlation(lagged_vals)

    # y_test_unnormalized = (ytest * data_std) + data_mean
    # y_pred_unnormalized = (ypred * data_std) + data_mean

    # trade_env = Trader(y_test_unnormalized.tolist(), y_pred_unnormalized.tolist(), writer, 18)
    # trade_env.trade()
    # result = trade_env.pay_back()
    # print ("tota profit", result)

    writer.close()
    
    print ("ypred: " , ypred)
    return ypred

def run_submission_session():
    while True:
        start = dt.now()
        bias = 0
        # calculate bias from previous result log
        result_csv = get_result_from_web()
        if result_csv:
            bias = get_bias(result_csv)
            print("bias value: ", bias)
        
        # make predictio
        pred_val = main(args)
        print("prediction value:", pred_val)
        
        final_val = pred_val + bias

        print("submitting answer as ", pred_val + bias)
        submit_answer(final_val)

        end = dt.now()

        # wait for re-submission
        elapsed = (end - start).seconds
        wait_time = RESUBMISSION_TIME_INTERVAL - elapsed

        print("WAITING FOR ...", wait_time , " ")
        time.sleep(wait_time)
        print("TIME: ", dt.now(), "Starting main()")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ai4impact project")

    parser.add_argument("--mode", type=str, default="main")
    parser.add_argument("--data",  action='store_true')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.prev = 9932

    # globals()[args.mode](args)
    run_submission_session()
    
    
    
    