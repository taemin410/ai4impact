from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import final_dataset
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
    wind_dataset = final_dataset()

    loader = DataLoader(
        wind_dataset,
        batch_size=8,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )

    split_ratio = 0.2
    batch_size = 16

    for epoch in range(num_epochs):
        # Train:   
        for batch_index, (faces, labels) in enumerate(train_loader):



        train_dataset, val_dataset, test_dataset = final_dataset(split_ratio, batch_size)

    # input_dim = wind_dataset[0:1][0].shape[1]

    loader = torch.utils.data.DataLoader(data, batch_size=8)

    # initialize Model
    model = NN_Model(
        input_dim=input_dim, output_dim=1, hidden_layers=modelConfig["hiddenlayers"]
    )

    model.train(loader, epochs=20)

    testX, testY = loader.dataset[1:100]
    print(testX.shape, testY.shape)

    rmse = model.test(testX, testY)
    print("RMSE:  ", rmse)

    # Evaluation phase
    # output = model.eval()

    # log(output)

    # visualize(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ai4impact project")

    parser.add_argument("--mode", type=str, default="main")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    globals()[args.mode](args)
        
