import torch
from torch.optim import Adam
import torch.nn.functional as F

from rlpd_pytorch.networks.actor import Actor
from rlpd_pytorch.networks.critic import Critic
from rlpd_pytorch.networks.temperature import Temperature


class SAC(object):
    def __init__(self, obs_shape, action_shape, device, hidden_dim, mlp_layers,
                 actor_lr, critic_lr, temp_lr, tau, discount, back_entropy, init_temp):
        super().__init__()
        self.tau = tau
        self.discount = discount
        self.back_entropy = back_entropy

        self.target_entropy = -action_shape[0] / 2

        self.actor = Actor(obs_shape[0], action_shape[0], hidden_dim, mlp_layers).to(device)
        self.critic = Critic(obs_shape[0], action_shape[0], hidden_dim, mlp_layers).to(device)

        self.critic_target = Critic(obs_shape[0], action_shape[0], hidden_dim, mlp_layers).to(device)
        self.temperatur = Temperature(init_temp).to(device)

        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.temp_optim = Adam(self.temperatur.parameters(), lr=temp_lr)

        self.hard_update(self.critic_target, self.critic)

    def update(self, ds, batch_size, utd_ratio):
        critic_losses = []
        actor_losses = []
        temp_losses = []
        for i in range(utd_ratio):
            def slice(d, batch_size, i):
                batch = {}

                for k, v in d.items():
                    batch[k] = v[i*batch_size:(i+1)*batch_size]
                return batch

            mini_batch = slice(ds, batch_size, i)

            critic_loss = self.update_critic(mini_batch)
            actor_loss = self.update_actor(mini_batch)
            temp_loss = self.update_temp(mini_batch)

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            temp_losses.append(temp_loss.item())

        return critic_losses, actor_losses, temp_losses

    def update_temp(self, mini_batch):
        action, log_prob, mean, mean_tanh = self.actor.sample(mini_batch['obs'])
        loss = self.temperatur() * (-log_prob.mean() - self.target_entropy).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        return loss

    def update_actor(self, mini_batch):
        action, log_prob, mean, mean_tanh = self.actor.sample(mini_batch['obs'])

        q1, q2 = self.critic(mini_batch['obs'], action)
        q = torch.cat([q1, q2], dim=1)
        q, _ = torch.min(q, dim=1) # 也可以是均值

        loss = (log_prob * self.temperatur() - q).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss

    def update_critic(self, mini_batch):
        with torch.no_grad():
            next_action, next_log_prob, next_mean, next_mean_tanh = self.actor.sample(mini_batch['next_obs'])
            next_t_q1, next_t_q2 = self.critic_target(mini_batch['next_obs'], next_action)
            next_t_q = torch.cat([next_t_q1, next_t_q2], dim=1)
            next_t_q, _ = torch.min(next_t_q, dim=1)

            if self.back_entropy:
                target_q = mini_batch['rewards'].unsqueeze(1) + self.discount * mini_batch['masks'].unsqueeze(1) \
                           * (next_t_q - self.temperatur() * next_log_prob)
            else:
                target_q = mini_batch['rewards'].unsqueeze(1) + self.discount * mini_batch['masks'].unsqueeze(1) * next_t_q

        q1, q2 = self.critic(mini_batch['obs'], mini_batch['actions'])
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        self.soft_update(self.critic_target, self.critic, self.tau) # 参数更新

        return q_loss

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
