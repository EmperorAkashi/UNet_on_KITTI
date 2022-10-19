from pyexpat import model
import torch
import numpy as np
from torch import nn
from sklearn.datasets import load_digits
import dataclasses

class FCN(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size):
        super().__init__()
        self.linear1 = nn.Linear(in_ch, filter_size)
        self.linear2 = nn.Linear(filter_size, filter_size)
        self.linear3 = nn.Linear(filter_size, out_ch)
        self.relu = nn.functional.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x

@dataclasses.dataclass
class run_config:
    model = FCN
    lr = 1e-5
    dev = 'cpu'

@dataclasses.dataclass
class train_config:
    num_epoch = 5


def train_one(data, _config):
    optimizer = torch.optim.Adam(_config.model.parameters(), lr = _config.lr)
    device = torch.device(_config.dev)

    cost = []
    for img, cls in data:
        optimizer.zero_grad()
        out = model(img.to(device))
        loss = nn.MSELoss(out, cls.to(device))
        loss.backward()
        optimizer.step()
        cost.append(loss.item)
    return cost

def train(model_config = run_config, train_config = train_config):
    cost_list = []
    for e in range(train_config.num_epoch):
        mini_cost = train(model_config.model, model_config.data_l, model_config.learning_rate)
        cost_list += mini_cost
    return cost_list

if __name__ == "__main__":
    train()


    



        