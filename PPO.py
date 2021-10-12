import torch.nn as nn
from tensorboardX import SummaryWriter

from network import ActorCritic
from rollout import RolloutBuffer
from config import *


class PPO:
    def __init__(self):

        if has_continuous_action_space:
            self.action_std = action_std

        self.buffer = RolloutBuffer()

        # 需要用到两个网络, 因为策略比
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([{
            'params': self.policy.actor.parameters(),
            'lr': lr_actor
        }, {
            'params': self.policy.critic.parameters(),
            'lr': lr_critic
        }])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 复制网络
        self.MSEloss = nn.MSELoss()

        self.writer = SummaryWriter(directory)
        # TODO 把critic和actor分开, 额外训练Ｑ值
        self.update_iteration = 0

    def set_action_std(self, new_action_std):
        if has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self):
        print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            state_value = self.policy.critic(state)

        if has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_value
        else:
            return action.item(), action_logprob, state_value

    # 通过sgd来更新模型, 一帧对应一次更新, k_epochs控制每一帧的重复利用率
    def update_sgd(self):
        if use_gae:
            returns, actions, logprobs, states = self.calc_gae_return()
        else:
            returns, actions, logprobs, states = self.calc_lambda_return()
        """trick2, reward normalizing"""
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # TODO, 改成running reward, 而且好像应该是给reward做norm而不是return

        # list 转 tensor
        old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(logprobs, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(device)

        # 进行k轮update policy
        for _ in range(k_epochs):
            for i in range(mini_batch):
                logprobs, state_values, entropy = self.policy.evaluate(old_states[i], old_actions[i])

                # 处理state_values的张量维度和reward相同
                state_values = torch.squeeze(state_values)

                # 计算策略比
                ratios = torch.exp(logprobs - old_logprobs[i].detach())

                # 计算PPO的约束loss
                advantages = returns[i] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2)
                """trick1, value function clipping"""
                # TODO value_clip的critic_loss也需要改成不求均值的形式
                if use_value_clip:
                    # 0.5就相当于epsilon, 是我瞎写的, 需要根据实际任务而定
                    _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
                    old_state_values = torch.squeeze(old_state_values)
                    value_clip = old_state_values + torch.clamp(state_values - old_state_values, -0.5, 0.5)
                    critic_loss = torch.min(self.MSEloss(state_values, returns), self.MSEloss(value_clip, returns))
                else:
                    critic_loss = (state_values - returns[i]).pow(2)

                # TODO entropy没有grad是不是有问题
                # 总的loss = actor loss + critic loss + entropy loss
                loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy
                if loss > 10000:
                    print("high loss! ", i, actor_loss, critic_loss, entropy, state_values, returns[i], advantages, surr1, surr2)
                self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.update_iteration)
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.update_iteration)
                self.writer.add_scalar('Loss/entropy', entropy, global_step=self.update_iteration)
                self.writer.add_scalar('Loss/total_loss', loss, global_step=self.update_iteration)

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                """trick9, global gradient clipping"""
                # max_grad_norm的值也是我瞎写的
                # nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                self.update_iteration += 1

        # 将新的权重赋值给policy old
        self.policy_old.load_state_dict(self.policy.state_dict())

    # 通过整个minibatch求平均来更新模型, 更准但效率低
    def update_minibatch(self):
        if use_gae:
            returns, actions, logprobs, states = self.calc_gae_return()
        else:
            returns, actions, logprobs, states = self.calc_lambda_return()
        """trick2, reward normalizing"""
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # TODO, 改成running reward, 而且好像应该是给reward做norm而不是return

        # list 转 tensor
        old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(logprobs, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(device)

        # 进行k轮update policy
        for _ in range(k_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(old_states, old_actions)

            # 处理state_values的张量维度和reward相同
            state_values = torch.squeeze(state_values)

            # 计算策略比
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算PPO的约束loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            """trick1, value function clipping"""
            # TODO value_clip的critic_loss也需要改成不求均值的形式, 同时考虑一下要不要改成SGD, 现在是ＭiniBatch

            if use_value_clip:
                # 0.5就相当于epsilon, 是我瞎写的, 需要根据实际任务而定
                _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
                old_state_values = torch.squeeze(old_state_values)
                value_clip = old_state_values + torch.clamp(state_values - old_state_values, -0.5, 0.5)
                critic_loss = torch.min(self.MSEloss(state_values, returns), self.MSEloss(value_clip, returns))
            else:
                critic_loss = (state_values - returns).pow(2)

            # TODO entropy没有grad是不是有问题
            # 总的loss = actor loss + critic loss + entropy loss
            loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy
            self.writer.add_scalar('Loss/critic_loss', critic_loss.mean(), global_step=self.update_iteration)
            self.writer.add_scalar('Loss/actor_loss', actor_loss.mean(), global_step=self.update_iteration)
            self.writer.add_scalar('Loss/entropy', entropy.mean(), global_step=self.update_iteration)
            self.writer.add_scalar('Loss/total_loss', loss.mean(), global_step=self.update_iteration)

            # 梯度更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            """trick9, global gradient clipping"""
            # max_grad_norm的值也是我瞎写的
            # nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.optimizer.step()

            self.update_iteration += 1

        # 将新的权重赋值给policy old
        self.policy_old.load_state_dict(self.policy.state_dict())

    # 计算MiniBatch的GAE, return = V + A(GAE)
    def calc_gae_return(self):
        actions, logprobs, states, next_states, states_value, rewards, is_done = self.buffer.sample()
        returns = []
        for t in range(self.buffer.mini_batch - 1, -1, -1):
            if is_done[t] or t == self.buffer.mini_batch - 1:  # episode最后一帧或者batch最后一帧
                gae = 0
                next_value = self.policy.critic(torch.FloatTensor(next_states[t]).to(device))
                delta = rewards[t] + gamma * next_value - states_value[t]
            else:
                delta = rewards[t] + gamma * states_value[t + 1] - states_value[t]
            gae += gamma * gae_lambda * delta
            returns.insert(0, gae + states_value[t])
        return returns, actions, logprobs, states

    # 计算MiniBatch的LambdaReturn, return = reward + gamma * return_next
    def calc_lambda_return(self):
        actions, logprobs, states, next_states, states_value, rewards, is_done = self.buffer.sample()
        returns = []
        for t in range(self.buffer.mini_batch - 1, -1, -1):
            lambda_return = 0
            if is_done[t]:  # 遇到done的时候重新计算
                lambda_return = 0
            lambda_return = rewards[t] + gamma * lambda_return
            returns.insert(0, lambda_return)
        return returns, actions, logprobs, states

    # 保存模型
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    # 加载模型
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
