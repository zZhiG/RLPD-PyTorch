import torch
import torch.nn as nn

from rlpd_pytorch.networks.mlp import MLP


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, mlp_layers=2):
        super().__init__()
        self.q1 = nn.Sequential(
            MLP(obs_shape + action_shape, hidden_dim, mlp_layers),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            MLP(obs_shape + action_shape, hidden_dim, mlp_layers),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, obs, action):
        input = torch.cat([obs, action], dim=1) # 当前的例子只有一维，当有多个环境或者其他情况时，需要更改拼接维度

        out1 = self.q1(input)
        out2 = self.q2(input)

        return out1, out2