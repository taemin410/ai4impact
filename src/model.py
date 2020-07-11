import torch.nn as nn
from torch.nn import functional as F  

class NN_Model(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        
        # Store input and output dimensions to instance var
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        current_input_dim = self.input_dim

        self.layers = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim = hidden_dim
        # add the last layer
        self.layers.append(nn.Linear(current_input_dim, self.output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)

        return out
