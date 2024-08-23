import torch
import torch.nn as nn
from torch.distributions import Normal

from rlpd_pytorch.networks.mlp import MLP


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, mlp_layers=2):
        super().__init__()
        self.base = MLP(obs_shape, hidden_dim, mlp_layers)

        self.mean = nn.Linear(hidden_dim, action_shape)
        self.log_std = nn.Linear(hidden_dim, action_shape)

    def forward(self, x):
        x = self.base(x)
        mean = self.mean(x)
        log_std = self.log_std(x)

        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)

        dist = Normal(mean, log_std.exp())  # 采用高斯分布

        x_t = dist.rsample()  # 采样动作
        action = torch.tanh(x_t)  # 固定到[-1, 1]
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        if len(log_prob.shape) == 2:
            log_prob = log_prob.sum(1, keepdim=True)  # 动作概率的log, 目前只有一个维度
        else:
            log_prob = log_prob.sum(0, keepdim=True)

        return action, log_prob, mean, torch.tanh(mean)
