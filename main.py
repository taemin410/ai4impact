from torch.utils.data import DataLoader
from src.dataset import *
# torch.utils.data.DataLoader

def main(args):

    # Load configurations 
    configs = load_config()

    # Preprocess the data
    split_ratio = 0.2
    wind_dataset = final_dataset(split_ratio)

    loader = DataLoader(wind_dataset, batch_size=8, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    data =  Data(configs)

    model = Model(configs)

    model.train()
    pickle_save()

    # Evaluation phase
    output = model.eval()

    log(output)



# visualize(output)



