import torch

from rlpd_pytorch.data.dataset import DataSet


class ReplayBuffer(DataSet):
    def __init__(self, obs_space, action_space, capacity, seed, device):
        data_dict = dict(
            obs=torch.empty((capacity, obs_space[0])).to(device),
            next_obs=torch.empty((capacity, obs_space[0])).to(device),
            rewards=torch.empty(capacity).to(device),
            actions=torch.empty((capacity, action_space[0])).to(device),
            masks=torch.empty(capacity).to(device),
            dones=torch.empty(capacity).to(device)
        ) # 根据任务而定

        super().__init__(data_dict, seed)
        self.capacity = capacity
        self.size = 0
        self.insert_index = 0

    def __len__(self):
        return self.size

    def insert(self, data_dict):
        # 传入为一个相同的字典
        assert data_dict.keys() == self.data_dict.keys(), '数据不一致！'

        for k, v in data_dict.items():
            self.data_dict[k][self.insert_index] = v

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)