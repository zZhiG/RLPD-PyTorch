import gym
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics

from rlpd_pytorch.wrappers.single_precision import SinglePrecision
from rlpd_pytorch.wrappers.universal_seed import UniversalSeed
from rlpd_pytorch.wrappers.to_pytorch import VecPyTorch

'''
包装器可以根据后续的具体任务和要求，再自定义相应的功能
同时可以增加多个环境，并行训练
'''

def wrap_gym(env: gym.Env, device, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env) # 单精度转换
    env = UniversalSeed(env) # 随机种子设置
    if rescale_actions:
        env = RescaleAction(env, -1, 1) # scale动作 ---> [-1, 1]

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = ClipAction(env)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env = VecPyTorch(env, device) # to pytorch

    return env
