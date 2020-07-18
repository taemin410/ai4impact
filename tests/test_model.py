from torch.utils.data import DataLoader

from src.model import NN_Model

import torch
import pytest
import numpy as np


def test_model():
    n = 64
    input_dim = 100

    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=1, size=(n, 1))

    data = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    )

    loader = torch.utils.data.DataLoader(data, batch_size=8)

    # initialize Model
    model = NN_Model(input_dim=input_dim, output_dim=1, hidden_layers=[16, 8])

    model.train(loader, epochs=20)

    testX, testY = loader.dataset[:50]
    print(testX.shape, testY.shape)

    rmse = model.test(testX, testY)
    print("RMSE:  ", rmse)
