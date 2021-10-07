import torch

from config import *


class RolloutBuffer:
    def __init__(self, mini_batch, batch_size):
        self.mini_batch = mini_batch  # 每次训练取出的数据量
        self.batch_size = batch_size  # PPO buffer总的数据存储大小
        self.cur_size = 0  # 当前buffer里的数据量

        self.actions = []  # 当前帧动作
        self.logprobs = []  # 当前帧动作的logp
        self.states = []  # 当前帧状态
        self.next_states = []  # 下一帧状态
        self.states_value = []  # 当前帧状态值函数
        self.rewards = []  # 当前帧奖励
        self.is_done = []  # 当前帧结束状态

    # 朝buffer里添加新数据
    def append(self, action, logprob, state, next_state, state_value, reward, done):
        if self.cur_size < self.batch_size:
            self.actions.append(torch.Tensor([action]).to(device))
            self.logprobs.append(logprob)
            self.states.append(torch.FloatTensor(state).to(device))
            self.next_states.append(next_state)
            self.states_value.append(state_value)
            self.rewards.append(reward)
            self.is_done.append(done)
            self.cur_size += 1
        else:
            self.actions = self.actions[1:] + [torch.Tensor([action]).to(device)]
            self.logprobs = self.logprobs[1:] + [logprob]
            self.states = self.states[1:] + [torch.FloatTensor(state).to(device)]
            self.next_states = self.next_states[1:] + [next_state]
            self.states_value = self.states_value[1:] + [state_value]
            self.rewards = self.rewards[1:] + [reward]
            self.is_done = self.is_done[1:] + [done]

    # 取出MiniBatch的数据并删除
    def sample(self):
        if self.cur_size < self.mini_batch:
            print("data is not enough,", self.cur_size, self.mini_batch)
            return
        # 取出数据
        actions = self.actions[:self.mini_batch]
        logprobs = self.logprobs[:self.mini_batch]
        states = self.states[:self.mini_batch]
        next_states = self.next_states[:self.mini_batch]
        states_value = self.states_value[:self.mini_batch]
        rewards = self.rewards[:self.mini_batch]
        is_done = self.is_done[:self.mini_batch]
        # # 删除数据
        # self.actions = self.actions[self.mini_batch:]
        # self.logprobs = self.logprobs[self.mini_batch:]
        # self.states = self.states[self.mini_batch:]
        # self.next_states = self.next_states[self.mini_batch:]
        # self.states_value = self.states_value[self.mini_batch:]
        # self.rewards = self.rewards[self.mini_batch:]
        # self.is_done = self.is_done[self.mini_batch:]
        # self.cur_size -= self.mini_batch
        return actions, logprobs, states, next_states, states_value, rewards, is_done

    # 判断buffer是否满
    def is_full(self):
        return self.cur_size == self.batch_size

    # 清空buffer
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.states_value[:]
        del self.rewards[:]
        del self.is_done[:]