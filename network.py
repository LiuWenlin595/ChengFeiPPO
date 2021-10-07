import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from config import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)
        """trick3, 正交初始化"""
        """trick8, tanh做激活函数"""
        if use_orth:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: m

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(init_(nn.Linear(state_dim, 64)), nn.Tanh(), init_(nn.Linear(64, 64)), nn.Tanh(), init_(nn.Linear(64, action_dim)),
                                       nn.Tanh())
        else:
            self.actor = nn.Sequential(init_(nn.Linear(state_dim, 64)), nn.Tanh(), init_(nn.Linear(64, 64)), nn.Tanh(), init_(nn.Linear(64, action_dim)),
                                       nn.Softmax(dim=-1))

        # critic
        self.critic = nn.Sequential(init_(nn.Linear(state_dim, 64)), nn.Tanh(), init_(nn.Linear(64, 64)), nn.Tanh(), init_(nn.Linear(64, 1)))

    def set_action_std(self, new_action_std):
        """设置方差"""
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim, ), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """actor, 输入状态, 输出动作"""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """actor + critic, 输入状态, 输出动作+动作值函数+熵"""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # 输出单动作的环境需要特殊处理一下数据
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
