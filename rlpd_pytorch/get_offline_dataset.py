import os
import numpy as np
import torch


def get_data(offline_data_path, device):
    files = os.listdir(offline_data_path) # 遍历文件夹下文件
    assert len(files) == 6, \
        '数据文件数量异常！' # 可根据具体任务和需要修改

    dicts = {} # 保存数据集元素和地址的键值对
    for _ in range(len(files)):
        name = files[_].split('.')[0]
        files[_] = os.path.join(offline_data_path, files[_]) # 获得完整路径
        dicts.update({name: files[_]})

    datasets = {}
    for k, v in dicts.items():
        d = np.load(v, ).astype(np.float32)
        d = torch.Tensor(d)
        datasets.update({k: d.to(device)})

    return datasets

def sample(data_dict, batchsize, seed):
    keys = list(data_dict.keys())
    len = data_dict[keys[0]].shape[0]

    np.random.seed(seed)
    start = np.random.randint(0, len - batchsize)  # 随机截取其中若干轨迹

    batch = {}

    for k in keys:
        batch[k] = data_dict[k][start:start + batchsize]

    return batch