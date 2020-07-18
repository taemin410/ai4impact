from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import final_dataset, load_dataset
from src.utils.config import load_config
from src.model import NN_Model
from src.trade.trader import Trader
from src.utils.logger import Logger
from datetime import datetime
import argparse
import torch
from datetime import datetime

def write_configs(writer, configs):
    configstr = ""
    for i in configs:
        configstr += str(i) + " : " + str(configs[i]) + "\n"
    writer.add_text("CONFIGS", configstr, 0)


def main(args):

    # Load configurations
    configs = load_config("config.yml")
    modelConfig = configs["model"]

    time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    logdir = "runs/" + time
    
    # Initialize SummaryWriter for tensorboard
    writer = Logger(logdir)
    write_configs(writer, modelConfig)
    # Initialize SummaryWriter for tensorboard
    # writer = SummaryWriter(logdir)
    # write_configs(writer, modelConfig)
  
    # Preprocess the data
    train_loader, validation_loader, test_loader, data_mean, data_std = load_dataset(
        batch_size=modelConfig["batchsize"]
    )

    # initialize Model
    model = NN_Model(
        input_dim=train_loader.dataset.tensors[0].size(1),
        output_dim=1,
        hidden_layers=modelConfig["hiddenlayers"],
        writer=writer,
    )

    model.train(
        train_loader,
        validation_loader,
        epochs=modelConfig["epochs"],
        lr=modelConfig["lr"],
    )

    rmse, ypred, ytest = model.test(test_loader)

    print("RMSE:  ", rmse)

    writer.add_text("RMSE", str(rmse.item()), 0)

    y_test_unnormalized = (ytest * data_std) + data_mean
    y_pred_unnormalized = (ypred * data_std) + data_mean

    print(y_test_unnormalized, y_pred_unnormalized)

    trade_env = Trader(y_test_unnormalized.tolist(), y_pred_unnormalized.tolist(), writer, 18)
    trade_env.trade()
    result = trade_env.pay_back()
    print (result)

    writer.close()

    # # Evaluation phase
    # # output = model.eval()

    # # log(output)

    # # visualize(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ai4impact project")

    parser.add_argument("--mode", type=str, default="main")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    globals()[args.mode](args)
