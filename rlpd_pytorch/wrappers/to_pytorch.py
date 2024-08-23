import torch
import gym
import numpy as np


class VecPyTorch(gym.Wrapper):
    def __init__(self, env, device):
        super(VecPyTorch, self).__init__(env)
        self.device = device

    def reset(self, **kwargs):
        observation = super(VecPyTorch, self).reset(**kwargs)
        obs = torch.from_numpy(np.array(observation)).float().to(self.device)

        return obs

    def step(self, action):
        observation, reward, done, info = super(VecPyTorch, self).step(action)
        observation = torch.from_numpy(np.array(observation)).float().to(self.device)
        reward = torch.from_numpy(np.array(reward)).float().to(self.device)

        return observation, reward, done, info
