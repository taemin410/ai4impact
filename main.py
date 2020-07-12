from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import final_dataset
from src.utils.config import load_config
from src.model import NN_Model

import argparse
import torch

def main(args):

    # Load configurations 
    configs = load_config('config.yml')
    modelConfig = configs["model"]
    trainConfig = configs["train"]

    # Initialize SummaryWriter for tensorboard
    writer = SummaryWriter()
    
    # Preprocess the data
    split_ratio = 0.2
    wind_dataset = final_dataset(split_ratio)

    loader = DataLoader(wind_dataset, batch_size=8, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    data =  Data(configs)

    # initialize Model 
    model = NN_Model(input_dim=modelConfig["hiddenlayers"], output_dim=1, hiddenlayers=modelConfig["hiddenlayers"])

    # model.train(writer=writer)
    # pickle_save()

    # Evaluation phase
    # output = model.eval()

    # log(output)

    # visualize(output)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ai4impact project")
    
    parser.add_argument("--mode", type=str, default='main')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)
