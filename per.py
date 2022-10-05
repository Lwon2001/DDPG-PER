import random

import numpy
import numpy as np
import torch

from sumtree import SumTree


class PrioritizedReplayBuffer:  # stored as ( s, a, r, s_ ,done) in SumTree
    def __init__(self, capacity, epsilon=0.01, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity  # buffer capacity
        self.epsilon = epsilon  # 一个很小的正数，用来防止当td-error为0的时候transition永远不再被选取
        self.alpha = alpha  # 决定用多少优先级（0-不使用优先级，1-使用全部优先级）
        self.beta = beta  # Anealing the bias 中的beta系数（需要从一个小于0的数逐渐增加为1）
        self.beta_increment_per_sampling = beta_increment  # 每次sample增加的beta值

    # 计算优先级p
    def compute_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    # 将transition与p加入优先级
    def add(self, error, sample):
        p = self.compute_priority(error)
        self.tree.add(p, sample)

    def get_transition_num(self):
        return self.tree.get_n_entries()

    # sample一组batch_size大小的数据
    def sample(self, batch_size):
        batch = []  # transition数据
        # random.seed(6)
        idxs = []  # 优先级保存在树中对应的idx
        priorities = []  # 优先级数据
        segment = self.tree.total() / batch_size  # 分成batchsize个区间（segment）

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)  # 每个segment中随机抽取
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()  # P(i)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()  # important-sampling weight

        return batch, idxs, is_weight

    # 更新优先级
    def update(self, idx, error):
        p = self.compute_priority(error)
        self.tree.update_priority(idx, p)
