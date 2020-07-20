import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import StepLR
import math

# def custom_loss(output, ytrue):

class NN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, writer):
        super().__init__()

        self.writer = writer

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
        x0 = x[:,4].clone().unsqueeze(1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)

        return out + x0

    def train(self, trainloader, validationloader, epochs=10, lr=0.01):

        # Initialize loss function and optimizer
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

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
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                valX = validationloader.dataset.tensors[0]
                valY = validationloader.dataset.tensors[1]

                outputs = self(valX).squeeze(1)
                val_loss = criterion(outputs, valY)
                self.writer.draw_validation_result(valY, outputs, epoch)
                
            self.writer.add_scalar("Loss/train", loss_sum / len(trainloader), epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            
            scheduler.step()

            if epoch % 1 == 0:
                print(
                    "Epoch: %d, batch loss: %1.5f Loss Sum: %1.5f"
                    % (epoch, loss.item(), loss_sum)
                )

        print("---train finished---")

    def test(self, test_loader):

        testX = test_loader.dataset.tensors[0]
        testY = test_loader.dataset.tensors[1]

        ypred = self(testX).squeeze(1)
        result = (testY - ypred) ** 2  # squared error

        rmse = (torch.sum(result) / result.shape[0]) ** 0.5  # root mean squared error
        for i in range(list(testY.size())[0]):
            self.writer.add_scalars(
                "test/pred", {"ypred": ypred[i], "ytrue": testY[i],}, i
            )

        return (rmse, ypred, testY)


class Persistance(nn.Module):
    def __init__(self, delay, writer):
        super().__init__()
        self.delay = delay
        self.writer = writer
    def forward(self, x):
        if self.delay == 0:
            return x
        return x[:, -self.delay]


    def test(self, test_loader):
        testX = test_loader.dataset.tensors[0]
        testY = test_loader.dataset.tensors[1]
        
        ypred = self(testX)
        result = (testY - ypred) ** 2  # squared error

        rmse = (torch.sum(result) / result.shape[0]) ** 0.5  # root mean squared error
        for i in range(list(testY.size())[0]):
            self.writer.add_scalars(
                "test/baseline", {"ypred": ypred[i], "ytrue": testY[i],}, i
            )

        return (rmse, ypred, testY)
