import os

# 当使用过高版本python，mujoco会报错，需要导入该
os.add_dll_directory(r"C:\Users\zhigao z\.mujoco\mujoco200\bin")
os.add_dll_directory(r"C:\Users\zhigao z\.mujoco\mujoco-py-master\mujoco-py-master\mujoco_py")

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import torch
import d4rl

from rlpd_pytorch.arguments import get_Configs
from rlpd_pytorch.wrappers import wrap_gym


def main():
    args = get_Configs()

    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = gym.make(args.env_name) # 测试环境创建
    env = wrap_gym(env, device, True)  # 包装
    env.seed(args.seed)

    weights = torch.load(args.weight_path, map_location='cuda:0')
    actor = weights['actor']
    critic = weights['critic']

    decode = {"r": "return", "l": "length", "t": "time"}

    for _ in range(args.eval_episodes):
        obs, done = env.reset(), False
        infomation = f'id:{_}, '
        while not done:
            env.render()
            action, _, _, _ = actor.sample(obs)
            action = action.detach().cpu().numpy()

            obs, reward, done, info = env.step(action)

        for k, v in info['episode'].items():
            infomation += f'{decode[k]}: {v}/ '

        print(f'{infomation}\n')


if __name__ == '__main__':
    main()