import torch
import torch.nn as nn


class Temperature(nn.Module):
    def __init__(self, init_temperature):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature), requires_grad=True)

    def forward(self):
        return self.temperature.exp()
