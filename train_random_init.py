import os
from datetime import datetime
import torch
import numpy as np
from multiprocessing import Process
import subprocess
import math

from PPO import PPO
from client import MyClient
from config import *


def obs_postprocess(raw_obs, mean_obs, std_obs):
    """trick6, 状态标准化; trick7, observation clipping"""
    """这个不实现, 因为每个环境的状态维度都不一样, 但是原理就是正规化, 即x = (x - x_mean) / x_std"""
    # 这里是数组操作, 因为不同特征的取值范围都不一样, 比如坐标是0-100, 伤害是0-10, 所以坐标和伤害的正规化要各自计算
    # new_obs = (raw_obs - mean_obs) / std_obs
    # 然后有一些数据正规化了之后还是过大过小, 需要再做一波clip
    # new_obs = np.clip(new_obs, -10, 10)
    raise NotImplementedError


# state的Z-zero归一化
def normalize_state(state):
    norm_state = np.zeros(state_dim)
    for i in range(state_dim):
        if state[i] == -1:  # 缺省值暂时设置为-1, 所以不需要做norm
            continue
        norm_state[i] = (state[i] - min_max[0]) / (min_max[1] - min_max[0])
    return norm_state


def update_linear_schedule(optimizer, timesteps, total_timesteps):
    """负责控制actor和critic学习率的线性衰减, 衰减力度都可以通过公式调"""
    ratio = 1 - timesteps / float(total_timesteps)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= ratio


def env_run():  # 注意！路径写死！
    p = subprocess.Popen(env_path, shell=True, stdout=subprocess.PIPE)
    p.communicate()


def env_close(old_env_proc):  # old_client
    subprocess.call("taskkill /F /T /PID {}".format(old_env_proc.pid))
    old_env_proc.terminate()
    # TODO 结束已有的client连接


def reset(old_env_proc):
    if old_env_proc != '':
        env_close(old_env_proc)
    p = Process(target=env_run, name='env_star')
    p.start()
    return p


# 设置随机种子
if random_seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def train():
    # 打印参数信息
    print_config()

    # 初始化PPO
    ppo_agent = PPO(state_dim, action_dim)
    if load_flag:
        print("--------------------------------------------------------------------------------------------")
        print("loading model at : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)
        print("model loaded")
        print("--------------------------------------------------------------------------------------------")

    # 记录训练时间
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # 开log的文件流
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # 定义 print/log average reward 的变量
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    client = MyClient()

    # 开始训练
    while time_step <= max_training_timesteps:
        """trick4, 学习率下降而非固定不变"""
        if use_linear_lr_decay:
            update_linear_schedule(ppo_agent.optimizer, time_step, max_training_timesteps)

        # print(datetime.now().replace(microsecond=0) - start_time)
        env_proc = reset(env_proc)  # 双端测试时注释掉
        client.send_reset()
        state = client.poll_reset()
        current_ep_reward = 0
        # print(datetime.now().replace(microsecond=0) - start_time)

        for t in range(max_ep_len):
            time_step += 1
            # 环境交互
            action_send = np.zeros(3)
            action_send[2] = state[2]
            norm_state = normalize_state(state)

            action, logprob, state_value = ppo_agent.select_action(norm_state)
            # action.shape = (1,)
            angle = ((state[5] + action[0] * 30) % 360) / 180.0 * math.pi
            action_send[0], action_send[1] = math.cos(angle), math.sin(angle)
            client.send_action(action_send)
            next_state, reward, done = client.recv_step(state)
            """trick5, reward clipping"""
            # reward = np.clip(reward, -5, 5)
            current_ep_reward += reward

            # buffer存一帧数据
            ppo_agent.buffer.append(action, logprob, state, next_state, state_value, reward, done)

            # 更新PPO
            if ppo_agent.buffer.is_full() and time_step % update_timestep == 0:
                ppo_agent.update()

            # 对于连续动作, 隔段时间降低动作标准差, 保证策略收敛
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log 记录 average reward
            if time_step % log_freq == 0:

                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # 打印 average reward
            if time_step % print_freq == 0:

                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # 保存模型
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done or t == max_ep_len - 1:  # 结束episode
                ppo_agent.writer.add_scalar('reward', current_ep_reward, global_step=i_episode)
                break

            state = next_state

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()

    # 打印完整的训练时间
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
