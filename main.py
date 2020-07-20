from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import final_dataset, load_dataset
from src.utils.config import load_config
from src.model import NN_Model, Persistance
from src.trade.trader import Trader
from src.utils.logger import Logger
from src.eval import get_lagged_correlation
from script.download_data import download_data, parse_data
from datetime import datetime
from src.request.request import submit_answer
import argparse
import torch
import threading
import time
from datetime import datetime

RESUBMISSION_TIME_INTERVAL = 600

def write_configs(writer, configs):
    configstr = ""
    for i in configs:
        configstr += str(i) + " : " + str(configs[i]) + "\n"
    writer.add_text("CONFIGS", configstr, 0)


def main(args):
    
    paths = download_data()
    for i in paths:
        parse_data(i)
    print("=============== Parsing dataset complete ===============")


    # Load configurations
    configs = load_config("config.yml")
    modelConfig = configs["model"]

    time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    logdir = "runs/" + time
    
    # Initialize SummaryWriter for tensorboard
    writer = Logger(logdir)
    write_configs(writer, modelConfig)
  
    # Preprocess the data
    train_loader, validation_loader, test_loader, data_mean, data_std = load_dataset(
        difference=0,
        batch_size=modelConfig["batchsize"]
    )
    
    # Baseline model
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

    ypred = model.predict(test_loader.dataset.tensors[0])
    y_pred_unnormalized  = (ypred * data_std) + data_mean

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
    
    print (y_pred_unnormalized)
    return y_pred_unnormalized

def run_submission_session():
    while True:
        pred_val = main(args)
        print("prediction value:", pred_val)

        submit_answer(pred_val)

        time.sleep(RESUBMISSION_TIME_INTERVAL)
        print("TIME: ", datetime.now(), "Starting main()")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ai4impact project")

    parser.add_argument("--mode", type=str, default="main")
    parser.add_argument("--data",  action='store_true')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # globals()[args.mode](args)
    run_submission_session()
    
    
    
    