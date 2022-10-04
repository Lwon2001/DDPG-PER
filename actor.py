import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):  # 定义Actor与Critic的网络结构
    def __init__(self, s_dim, a_dim, fc1_dim, fc2_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)

        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)

        self.out = nn.Linear(fc2_dim, a_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))  # 激活函数为relu
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.out(x)
        x = torch.tanh(x)  # 利用tanh将值映射到[-1,1]，因为该游戏的动作取值范围为[-1，1]
        return x
