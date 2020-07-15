from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import final_dataset, load_dataset
from src.utils.config import load_config
from src.model import NN_Model

import argparse
import torch


def main(args):

    # Load configurations
    configs = load_config("config.yml")
    modelConfig = configs["model"]
    trainConfig = configs["train"]

    # Initialize SummaryWriter for tensorboard
    # writer = SummaryWriter()

    # Preprocess the data
    train_loader, validation_loader, test_loader = load_dataset(batch_size=16)
        
    # initialize Model
    model = NN_Model(
        input_dim=299, output_dim=1, hidden_layers=modelConfig["hiddenlayers"]
    )

    model.train(train_loader, validation_loader, epochs=1)

    rmse = model.test(test_loader)
    print("RMSE:  ", rmse)

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
        
