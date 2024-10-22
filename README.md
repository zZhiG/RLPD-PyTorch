# <p align="center">RLPD-PyTorch</p>

## Introduction
This is a reproduction of an excellent work by ICML 2023, the work proposes the use of off-policy method and offline data in the online learning. It is very meaningful, as the idea can be applied to many scenarios.

Therefore, we modified and refactored its code, building a simple framework. A simple test was conducted on one of the environments, and it worked. Many of these parts may not be perfect, as we hope to use the framework in specific tasks in the future. So many places are not flexible enough and only provide simple examples. (For example, you can customize the environment, add different wrappers, modify networks, etc. to adapt to different tasks.)

For a more detailed description, you can read the original paper and source code.

If this is helpful to you, could you give me a star :star:.

## Show
We trained and validated the framework on ```HalfCheetah-v2``` to be effective:

<div align=center>
<img src="demo/demo001.gif" width="600px">
</div>

## Quick start
You can use the default configuration by running the ```train.py``` directly. Of course, you can also change the configuration manually.
```
python train.py
```

Also provided is a validated test code:
```
python eval.py --weight-path='your path'
```
## Dependencies
```
numpy == 1.26.4
gym == 0.17.0
d4rl == 1.1
python == 3.10
tensorboardx == 2.6.2.2
torch == 2.2.2+cu118
mujoco-py == 1.50.1.0
```

## Reference
original paper: [Efficient online reinforcement learning with offline data](https://dl.acm.org/doi/abs/10.5555/3618408.3618475)

source code: [RLPD](https://github.com/ikostrikov/rlpd)
