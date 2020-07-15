import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import dataloader


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
        
        print(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)

        return out

    def train(self, trainloader, validationloader, epochs=10, lr=0.01, writer=None):

        # Initialize loss function and optimizer
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        val_losses = []

        # Train the model by epochs
        for epoch in range(epochs):
            loss_sum = 0

            for xx, yy in trainloader:

                optimizer.zero_grad()
                outputs = self(xx)

                outputs = outputs.squeeze(1)
                
                # obtain the loss function
                # TODO: Add tensorboard write
                loss = criterion(outputs, yy)
                loss_sum += loss.clone()
                loss.backward()
                optimizer.step()
            
            losses.append(loss_sum)
            # with torch.no_grad():
            #     outputs = self(validationloader.dataset[0:100][0]).squeeze(1)
            #     val_loss = criterion(outputs, validationloader.dataset[0:100][1].squeeze(1))
            #     val_losses.append(val_loss)

            if epoch % 5 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        print("---train finished---")

    def test(self, test_loader):
        
        testX = test_loader.dataset.tensors[0]
        testY = test_loader.dataset.tensors[1]

        ypred = self(testX).squeeze(1)
        result = (testY - ypred) ** 2  # squared error

        rmse = (torch.sum(result) / result.shape[0]) ** 0.5  # root mean squared error

        return (rmse, ypred)


class Persistance(nn.Module):

    def __init(self, delay):
        super().init()
        self.delay = delay

    def forward(self, x):
        if self.delay == 0:
            return x
        return x[:,-self.delay]

