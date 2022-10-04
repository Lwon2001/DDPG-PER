import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim, fc1_dim, fc2_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(a_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, s, a):
        x_s = F.relu(self.ln1(self.fc1(s)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(a)
        x = F.relu(x_s + x_a)
        q = self.q(x)

        return q
