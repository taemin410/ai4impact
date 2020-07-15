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

    n = 300
    input_dim = 300
    import numpy as np

    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))

    # input_dim = wind_dataset[0:1][0].shape[1]

    data = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    )

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
