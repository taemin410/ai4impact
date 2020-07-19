import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import StepLR
import math

class NN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, writer, device):
        super().__init__()

        self.writer = writer
        self.device = device

        # Store input and output dimensions to instance var
        self.input_dim = input_dim
        self.output_dim = output_dim

        current_input_dim = self.input_dim

        #hidden layers
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim = hidden_dim

        # add the last layer
        self.layers.append(nn.Linear(current_input_dim, self.output_dim))

        # dropout
        # self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, x):
        # x = x.to(self.device)
        x0 = x[:,4].unsqueeze(1).clone()
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)

        return out + x0

    def train(self, trainloader, validationloader, epochs=10, lr=0.01, step_size=25, gamma=0.1, weight_decay=0):

        # Initialize loss function and optimizer
        criterion = torch.nn.SmoothL1Loss()  # mean-squared error for regression
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Train the model by epochs
        for epoch in range(epochs):
            loss_sum = 0
            val_loss_sum = 0
            for xx, yy in trainloader:
                # xx = xx.to(self.device)
                # yy = yy.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(xx)

                outputs = outputs.squeeze(1)

                # obtain the loss function
                # TODO: Add tensorboard write
                loss = self.log_cosh_loss_func(outputs, yy)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for valX, valY in validationloader:

                    # valX = valX.to(self.device)
                    # valY = valY.to(self.device)

                    outputs = self(valX).squeeze(1)
                    val_loss = self.log_cosh_loss_func(outputs, valY)
                    val_loss_sum += val_loss
                    # self.writer.draw_validation_result(valY, outputs, epoch)
                
            self.writer.add_scalar("Loss/train", loss_sum / len(trainloader), epoch)
            self.writer.add_scalar("Loss/validation", val_loss_sum / len(validationloader), epoch)
            
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

    def careful_predict_loss_func(self, pred, target):
        loss = 0
        # pred = pred.to(self.device)
        # target = target.to(self.device)

        for pred_val, target_val in zip(pred, target):
            diff = pred_val - target_val
            if abs(diff) > 1:
                loss += abs(diff) - 0.5
            else:
                if pred_val > target_val:   # buy energy with higher price -> penaltize more than usual
                    loss += 2 * (diff ** 2)
                else: 
                    loss += 0.5 * (diff ** 2)
        
        return loss / len(pred)

    def log_cosh_loss_func(self, pred, target):
        ey_t = pred - target
        return torch.mean(torch.log(torch.cosh(ey_t)))


class Persistance(nn.Module):
    def __init(self, delay):
        super().init()
        self.delay = delay

    def forward(self, x):
        if self.delay == 0:
            return x
        return x[:, -self.delay]
