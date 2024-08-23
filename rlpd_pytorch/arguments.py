import argparse
import torch

def get_Configs():
    parser = argparse.ArgumentParser(description='rlpd_pytorch') # 重构为pytorch版本
    parser.add_argument('--env-name', type=str, default='halfcheetah-expert-v2', help='D4rl dataset name')  # 环境名称 "HalfCheetah-v2"
    parser.add_argument('--offline-ratio', type=float, default=0.5, help='Offline ratio')  # offline数据所占比例
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  # 随机种子
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes used for evaluation')  # 评估时将测试多少周期
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval')  # 训练时，多少次保存一次日志
    parser.add_argument('--eval-interval', type=int, default=5000, help='Eval interval')  # 训练时，多少次评估一次
    parser.add_argument('--batch-size', type=int, default=256, help='Mini batch size')  # 训练时，每次采样多少数据
    parser.add_argument('--utd-ratio', type=int, default=2, help='Number of training steps')  # 开始更新的次数
    parser.add_argument('--max-steps', type=int, default=1e6, help='Update to data ratio')  # 更新多少组batch
    parser.add_argument('--start-training', type=int, default=1e4, help='Number of training steps')  # 开始更新的次数
    parser.add_argument('--pretrain-steps', type=int, default=1, help='Number of offline updates')  # 使用offline数据提前更新的次数
    parser.add_argument('--tqdm', type=bool, default=True, help='Use tqdm progress bar')  # 是否使用进度条
    parser.add_argument('--checkpoint-model', type=bool, default=True, help='Save agent checkpoint on evaluation')  # 是否保存权重
    parser.add_argument('--checkpoint-buffer', type=bool, default=False, help='Save agent replay buffer on evaluation')  # 是否保存数据
    parser.add_argument('--offline-data-path', type=str, default='data/halfcheetah_v2')  # 直接给出offline数据集地址
    parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda
    parser.add_argument('--mlp-layers', type=int, default=2)  # mlp的层数
    parser.add_argument('--weight-path', type=str, default='log/seed42_pretrain1_LN_steps1e+06_batch256x2/2024.08.23.12.06.58/checkpoints/454999.pt') # 测试权重路径

    # agent相关设置
    parser.add_argument('--actor-lr', type=float, default=3e-4)  # actor的学习率
    parser.add_argument('--critic-lr', type=float, default=3e-4)  # critic的学习率
    parser.add_argument('--temp-lr', type=float, default=3e-4)  # temperature的学习率
    parser.add_argument('--hidden-dim', type=int, default=256)  # 隐藏层维度
    parser.add_argument('--discount', type=float, default=0.99)  # 折扣因子
    parser.add_argument('--tau', type=float, default=0.005)  # critic更新率
    parser.add_argument('--init-temp', type=float, default=1.)  # 初始化temp，用来控制探索度
    parser.add_argument('--backup-entropy', type=bool, default=True)  # 是否考虑熵

    args = parser.parse_args()

    args.cuda = True if torch.cuda.is_available() else False

    return args