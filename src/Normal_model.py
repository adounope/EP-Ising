import numpy as np
import torch
import torch.nn as nn


class Normal_model(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=196, lr = 1):
        super(Normal_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lr = lr
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
    def train(self, x, y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, ((y>0) != (y_pred > 0)).sum().float() / y[0].numel()
    
    def label_to_num(self, y_raw):
        tmp = y_raw.reshape(len(y_raw), 40//4, -1).mean(axis=-1) #batch, 10
        tmp = (tmp == np.max(tmp, axis=1)[:, None]) #batch, 10 (bool)
        invalid = (tmp.sum(axis=1) != 1) # batch (detect case of multiple maximum)
        y = np.argmax(tmp, axis=1)
        y[invalid] = -1
        return y
