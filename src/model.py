import torch
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
 
    def train(self, dataset, epochs=10, batch_size=8, lr=0.001, writer=None):

        # Initialize loss function and optimizer 
        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        losses = []
        val_losses = [] 
        
        # Train the model by epochs
        for epoch in range(epochs):
            loss_sum = 0 
            for i in range(dataset[0].shape[0]//batch_size):
                optimizer.zero_grad()
                outputs = self(dataset[i*batch_size: (i+1)* batch_size][0]).squeeze(1)

                # obtain the loss function
                # TODO: find train_y from dataset \
                # TODO: Add tensorboard write 
                loss = criterion(outputs, dataset[i*batch_size: (i+1)* batch_size][1])
                loss_sum += loss.clone()
                loss.backward()
                optimizer.step()
            losses.append(loss_sum)

            with torch.no_grad():
                outputs = model(val_x).squeeze(1)
                # val_loss = (torch.sum((outputs - val_y)**2) / outputs.shape[0])**0.5
                val_loss = criterion(outputs, val_y)
                val_losses.append(val_loss)
                
        if epochs % 5 == 0:
            print("Epoch: %d, loss: %1.5f" % (epochs, loss.item()))

