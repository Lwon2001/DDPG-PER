import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from ddpg import DDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
EP_STEPS = 200

RENDER = False
ENV_NAME = 'LunarLanderContinuous-v2'

# 配置gym
env = gym.make(ENV_NAME)
env = env.unwrapped
env.reset(seed=1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]  # 动作值为[a1,a2]，a1控制油门，a2控制左右点火引擎，取值范围都为[-1,1]的实数
a_bound = env.action_space.high
a_low_bound = env.action_space.low


ddpg = DDPG(s_dim=s_dim, a_dim=a_dim, actor_fc1_dim=64, actor_fc2_dim=32, critic_fc1_dim=64, critic_fc2_dim=32)

var = 3  # 加入噪声用到的正态分布中的标准差
t1 = time.time()
reward_list = []

T = 20000
N = 100

for t in range(T):
    s = env.reset()
    ep_r = 0  # 每一个episode的累积奖励值
    for i in range(N):
        if RENDER: env.render()
        a = ddpg.get_action(s)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)  # 加入噪声
        s_, r, done, _, info = env.step(a)
        transition = (s, a, r , s_, done)
        ddpg.store_experience(1, transition)  # 存储与环境互动经验
        if ddpg.get_experience_num() > 200:  # 存储200个transition后开始训练
            var *= 0.9999  # decay the exploration controller factor
            ddpg.train()
        s = s_
        ep_r += r
        if i == N - 1:
            reward_list.append(ep_r)
            print('Episode: ', t, ' Reward: %i' % ep_r, 'Explore: %.2f' % var)

    if t > 0 and t % 100 == 0:
        ddpg.save_model()
        x = range(0, t + 1)
        plt.plot(x, reward_list, '.-')
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()
print('Running time: ', time.time() - t1)
