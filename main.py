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


def main(args):

    # Load configurations
    configs = load_config("config.yml")
    modelConfig = configs["model"]

    # Initialize SummaryWriter for tensorboard
    writer = Logger(datetime.now())

    # Preprocess the data
    train_loader, validation_loader, test_loader = load_dataset(batch_size=modelConfig["batchsize"])
        
    # initialize Model
    model = NN_Model(
        input_dim=299, output_dim=1, hidden_layers=modelConfig["hiddenlayers"], writer=writer
    )

    model.train(train_loader, validation_loader, epochs=modelConfig["epochs"], lr=modelConfig["lr"])

    rmse, ypred, ytest = model.test(test_loader)
    print("RMSE:  ", rmse)

    trade_env = Trader(ytest.tolist(), ypred.tolist(), writer, 18)
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
        
