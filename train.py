import copy
import os

# 当使用过高版本python，mujoco会报错，需要导入该
os.add_dll_directory(r"C:\Users\zhigao z\.mujoco\mujoco200\bin")
os.add_dll_directory(r"C:\Users\zhigao z\.mujoco\mujoco-py-master\mujoco-py-master\mujoco_py")

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import time
import gym
import d4rl
from tensorboardX import SummaryWriter
import torch
import tqdm
import numpy as np

from rlpd_pytorch.arguments import get_Configs
from rlpd_pytorch.wrappers import wrap_gym
from rlpd_pytorch.get_offline_dataset import get_data, sample
from rlpd_pytorch.agents.sac import SAC
from rlpd_pytorch.data.repalybuffer import ReplayBuffer


def combine(one_dict, other_dict, device):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = torch.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            ).to(device)
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined

def main():
    args = get_Configs()
    assert args.offline_ratio >= 0. and args.offline_ratio <= 1., \
        'offline数据的比例必须在0~1之间！'

    prefix = 'seed{}_pretrain{}_LN_steps{:.0e}_batch{}x{}'.\
        format(args.seed, args.pretrain_steps, args.max_steps, args.batch_size, args.utd_ratio) # 默认都采用LN

    timeStr = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))

    logdir = os.path.join('log', prefix, timeStr)

    if args.checkpoint_model:
        chkpt_dir = os.path.join(logdir, "checkpoints")
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

    if args.checkpoint_buffer:
        buffer_dir = os.path.join(logdir, "buffers")
        if not os.path.exists(buffer_dir):
            os.makedirs(buffer_dir)

    tb = os.path.join(logdir, "tb")
    writer = SummaryWriter(logdir=tb)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = gym.make(args.env_name) # 训练环境创建 "HalfCheetah-v2"
    env = wrap_gym(env, device, True) # 包装
    env.seed(args.seed)

    eval_env = gym.make(args.env_name) # 测试环境创建
    eval_env = wrap_gym(eval_env, device, True)
    eval_env.seed(args.seed * 2)

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    '''
    加载数据集，offline数据为事先准备好的，直接提供路径加载，
    包含obs，action，next_obs，reward，done，mask
    '''
    off_ds = get_data(args.offline_data_path, device) # 获得离线数据集，并转换为tensor

    agent = SAC(obs_shape, action_shape, device, args.hidden_dim, args.mlp_layers,
                args.actor_lr, args.critic_lr, args.temp_lr, args.tau, args.discount,
                args.backup_entropy, args.init_temp)

    replaybuffer = ReplayBuffer(obs_shape, action_shape, int(args.max_steps), args.seed, device)

    update_num = 0

    for i in tqdm.tqdm(range(0, args.pretrain_steps), smoothing=0.1, disable=not args.tqdm):
        ds = sample(off_ds, args.batch_size * args.utd_ratio, args.seed) # 采样

        critic_losses, actor_losses, temp_losses = agent.update(ds, args.batch_size, args.utd_ratio) # 更新
        for i in range(len(critic_losses)):
            writer.add_scalar(f'Traing/critic loss', critic_losses[i], update_num)
            writer.add_scalar(f'Traing/actor loss', actor_losses[i], update_num)
            writer.add_scalar(f'Traing/temp loss', temp_losses[i], update_num)
            # print(f'{update_num}: critic loss:{critic_losses[i]}, actor loss:{actor_losses[i]}, temp loss:{temp_losses[i]}')
            update_num += 1
    print('预训练结束')

    print('正式训练开始')
    obs, done = env.reset(), False

    for i in tqdm.tqdm(range(0, int(args.max_steps) + 1), smoothing=0.1, disable=not args.tqdm):
        if i < args.start_training:
            action = env.action_space.sample() # 随机采集动作
            # print(f'random:{action}')
        else:
            action, _, _, _ = agent.actor.sample(obs)
            # print(f'sample:{action}')

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
            next_obs, reward, done, info = env.step(action)
            action = torch.from_numpy(action).float().to(device)
        else:
            next_obs, reward, done, info = env.step(action)
            action = torch.from_numpy(action).float().to(device)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replaybuffer.insert(
            dict(
                obs=obs,
                actions=action,
                rewards=reward,
                dones=done,
                masks=mask,
                next_obs=next_obs
            )
        ) # 在线收集数据

        obs = next_obs

        if done:
            infomation = f'Training {i + args.pretrain_steps}/ '
            obs, done = env.reset(), False

            decode = {"r": "return", "l": "length", "t": "time"}
            for k, v in info['episode'].items():
                writer.add_scalar(f'Training/{decode[k]}', v, i + args.pretrain_steps)
                infomation += f'{decode[k]}: {v}/ '
            print('\n')

        if i >= args.start_training:
            online_batch = replaybuffer.sample(int(args.batch_size * args.utd_ratio * (1 - args.offline_ratio))) # 收集在线数据
            offline_batch = sample(off_ds, int(args.batch_size * args.utd_ratio * args.offline_ratio), args.seed) # 收集离线数据

            batch = combine(offline_batch, online_batch, device)
            critic_losses, actor_losses, temp_losses = agent.update(batch, args.batch_size, args.utd_ratio)  # 更新
            for j in range(len(critic_losses)):
                writer.add_scalar(f'Traing/critic loss', critic_losses[j], update_num)
                writer.add_scalar(f'Traing/actor loss', actor_losses[j], update_num)
                writer.add_scalar(f'Traing/temp loss', temp_losses[j], update_num)
                # print(f'{update_num}: critic loss:{critic_losses[i]}, actor loss:{actor_losses[i]}, temp loss:{temp_losses[i]}')
                update_num += 1

        if args.checkpoint_model and (i + 1) % args.eval_interval == 0:
            checkpoint = {
                "critic": agent.critic,
                'actor': agent.actor,
            }
            print(f'--保存权重：{i}.pt--')
            torch.save(checkpoint, os.path.join(chkpt_dir, f'{i}.pt')) # 保存critic和actor

        # 取消了训练过程中的评估，后期针对任务需要可以再补上
        # 目前只实现一个利用offline和online数据，采用off-policy方式训练的RL算法的 PyTorch版本大致框架
        # 删减和修改了原论文代码
        # 针对实际任务和问题需求，再进行针对性修改
        # 该框架在示例环境中能够成功训练，并完成任务，是有效的

if __name__ == '__main__':
    main()