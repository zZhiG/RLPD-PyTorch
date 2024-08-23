import torch
from typing import Optional, Iterable
import numpy as np

class DataSet(object):
    def __init__(self, data_dict, seed):
        super().__init__()
        self.data_dict = data_dict
        self.data_len = self.check_len(data_dict)
        self.seed = seed

    def check_len(self, data_dict):
        # 检测字典中每一个值的长度是否相等，类型是否正确
        init_len = 0
        for _, (k, v) in enumerate(data_dict.items()):
            if not isinstance(v, torch.Tensor):
                raise TypeError('数据类型错误，必须为tensor')

            if _ == 0:
                    init_len = v.shape[0]
            else:
                len = v.shape[0]
                assert len == init_len, '数据长度不相等'

        return init_len

    def set_seed(self, seed):
        self.seed = seed

    def __len__(self):
        return self.data_len

    def sample(self, batchsize,
               keys: Optional[Iterable[str]] = None):

        np.random.seed(self.seed)
        start = np.random.randint(0, self.__len__() - batchsize) # 随机截取其中若干轨迹

        batch = {}

        if keys is None:
            keys = self.data_dict.keys()

        for k in keys:
            batch[k] = self.data_dict[k][start:start+batchsize]

        return batch
