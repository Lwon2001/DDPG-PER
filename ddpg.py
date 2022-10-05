import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from actor import ActorNet
from critic import CriticNet
from per import PrioritizedReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# paper "Continuous control with deep reinforcement learning"

class DDPG(object):
    def __init__(self, s_dim, a_dim, actor_fc1_dim, actor_fc2_dim, critic_fc1_dim, critic_fc2_dim):
        # 构建 actor 和target actor
        self.actor = ActorNet(s_dim, a_dim, actor_fc1_dim, actor_fc2_dim).to(device)
        self.actor_target = ActorNet(s_dim, a_dim, actor_fc1_dim, actor_fc2_dim).to(device)

        # 初始化target actor的参数与actor参数相同
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 采用Adam进行优化，学习率采用DDPG paper中用的学习率
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # 构建 critic 和target critic
        self.critic = CriticNet(s_dim, a_dim, critic_fc1_dim, critic_fc2_dim).to(device)
        self.critic_target = CriticNet(s_dim, a_dim, critic_fc1_dim, critic_fc2_dim).to(device)

        # 初始化target critic 网络的参数与critic网络参数相同
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 采用Adam进行优化，L2 权重系数采用DDPG paper中的所用到的权重
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        # 采用Prioritized Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000, epsilon=0.01, alpha=0.6, beta=0.4,
                                                     beta_increment=0.001)

    # 根据状态作出决策
    def get_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        return self.actor(s).cpu().data.numpy().flatten()

    def save_model(self):
        torch.save(self.actor.state_dict(), 'weight/actor_weights.pth')
        torch.save(self.critic.state_dict(), 'weight/critic_weights.pth')

    def store_experience(self, td_error, transition):
        self.replay_buffer.add(td_error, transition)

    def get_experience_num(self):
        return self.replay_buffer.get_transition_num()

    # 训练，这里为了与之前的ddpg代码进行比较，将batch_size设置为1
    def train(self, prioritized=True, batch_size=1, gamma=0.99, tau=0.005):
        # Sample replay buffer
        transitions, idxs, is_weights = self.replay_buffer.sample(batch_size)
        transitions = np.array(transitions).transpose()
        states = np.vstack(transitions[0])
        actions = np.array(list(transitions[1]))
        rewards = np.array(list(transitions[2])).reshape(-1, 1)
        next_states = np.vstack(transitions[3])
        dones = transitions[4].reshape(-1, 1)

        # 数据处理，转换为tensor
        dones = dones.astype(int)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = rewards.reshape(-1, 1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        dones = dones.reshape(-1, 1)
        is_weights = torch.FloatTensor(is_weights).to(device)
        is_weights = is_weights.reshape(-1, 1)

        # 计算Q_target
        Q_target = self.critic_target(next_states, self.actor_target(next_states))

        # 计算 Y
        Y = (rewards + (1 - dones) * gamma * Q_target).detach()

        # 计算Q
        Q = self.critic(states, actions)

        # 计算TD-error
        TD_errors = (Y - Q)

        # 利用IS_weight对TD_error进行对应的衰减
        weighted_TD_errors = torch.mul(TD_errors, is_weights)
        weighted_TD_errors = weighted_TD_errors.to(device)

        # 计算critic_loss
        zero_tensor = torch.zeros(weighted_TD_errors.shape)
        zero_tensor = zero_tensor.to(device)
        critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # softly update
        for critic_weights, critic__target_weights in zip(self.critic.parameters(),
                                                          self.critic_target.parameters()):
            critic__target_weights.data.copy_(tau * critic_weights.data + (1 - tau) * critic__target_weights.data)
        for actor_weights, actor__target_weights in zip(self.actor.parameters(), self.actor_target.parameters()):
            actor__target_weights.data.copy_(tau * actor_weights.data + (1 - tau) * actor__target_weights.data)

        # 用计算出来的TD-error更新transition的优先级
        td_errors = TD_errors.detach().cpu().numpy()
        self.replay_buffer.update(idxs, td_errors)

