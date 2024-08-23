import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_layers=2, act=nn.ReLU, norm=nn.LayerNorm, dropout_rate=None):
        super().__init__()
        self.hidden_size = hidden_dim

        self.act = act()
        self.norm = norm(hidden_dim)
        if dropout_rate is not None and dropout_rate >= 0. and dropout_rate <= 1.:
            self.drop = nn.Dropout(dropout_rate)
        else:
            self.drop = nn.Identity()

        self.linears = nn.ModuleList([])
        for i in range(mlp_layers):
            if i == 0:
                self.linears.append(nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    self.act,
                    self.norm,
                    self.drop
                ))
            else:
                self.linears.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act,
                    self.norm,
                    self.drop
                ))

    def get_hidden_dim(self):
        return self.hidden_size

    def forward(self, x):
        for m in self.linears:
            x = m(x)

        return x
