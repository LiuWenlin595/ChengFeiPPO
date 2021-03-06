import math
import numpy as np
import random
from multiprocessing import Process
import subprocess
from enum import Enum

from config import *


# 对局结束指标
class Done(Enum):
    not_done = 0            # 对局继续
    time_out = 1            # 超过最大步长
    red_crash = 2           # 红色被摧毁
    blue_crash = 3          # 蓝色被摧毁, 暂时视为对局结束
    arrive_goal = 4         # 到达目标点
    self_crash_angle = 5    # 意外结束, 判断为拐角过大
    self_crash_coord = 6    # 意外结束, 判断为坐标偏离范围


# 计算reward和done
def get_reward_done(cur_state, next_state, red_crash, blue_crash):
    reward, done = 0, Done.not_done.value

    # 时间步小惩罚, 可以督促agent快点完成任务
    # reward -= 0.005  # 0.005x1000=5

    # 根据目标点距离给予连续性小奖励
    goal_lat, goal_lon, goal_height = cur_state[11], cur_state[12], cur_state[13]
    cur_dist = math.sqrt(pow(cur_state[0] - goal_lat, 2) + pow(cur_state[1] - goal_lon, 2))
    # next_dist = math.sqrt(pow(next_state[0] - goal_lat, 2) + pow(next_state[1] - goal_lon, 2))
    # reward += (math.exp(-next_dist) - math.exp(-cur_dist)) * 1000 # 采用y=e^(-x)作为距离的递减函数, 相比于线性函数可以更好的体现目标远近

    # TODO 根据导弹的范围来设计奖励, 但是导弹并不是全局信息, 所以需要考虑疏忽的情况

    if red_crash:   # 红方被击中, 给予一次性大惩罚
        print("red crash!")
        reward -= 3
        done = Done.red_crash.value
    elif next_state[0] < min_max[0][0] or next_state[0] > min_max[0][1] or next_state[1] < min_max[1][0] or \
            next_state[1] > min_max[1][1] or next_state[2] < min_max[2][0] or next_state[2] > min_max[2][1]:
        print("crash coord! ", next_state[:3])
        reward -= 20
        done = Done.self_crash_coord.value
    elif cur_dist < 0.1:    # 到达目标点给予一次性大奖励
        print("arrive goal!")
        reward += 50
        done = Done.arrive_goal.value
    elif blue_crash:    # 蓝方被击中, 给予一次性大奖励
        print("blue crash!")
        reward += 3
        done = Done.blue_crash.value

    # 坠机, 给予一次性大惩罚
    # delta_phi = abs(next_state[3] - cur_state[3])
    # if delta_phi > 180:
    #     delta_phi = 360 - delta_phi
    # delta_theta = abs(next_state[4] - cur_state[4])
    # if delta_theta > 180:
    #     delta_theta = 360 - delta_theta
    # delta_psi = abs(next_state[5] - cur_state[5])
    # if delta_psi > 180:
    #     delta_psi = 360 - delta_psi
    # if delta_phi > 15 or delta_theta > 15 or delta_psi > 15:
    #     print("crash down by angle! ", delta_phi, delta_theta, delta_psi, next_state[3], cur_state[3], next_state[4],
    #           cur_state[4], next_state[5], cur_state[5])
    #     reward -= 5
    #     done = Done.self_crash_angle.value
    # if next_state[0] < min_max[0][0] or next_state[0] > min_max[0][1] or next_state[1] < min_max[1][0] \
    #         or next_state[1] > min_max[1][1] or next_state[2] < min_max[2][0] or next_state[2] > min_max[2][1]:
    #     print("crash down by coord! ", next_state[0], next_state[1], next_state[2])
    #     reward -= 5
    #     done = Done.self_crash_coord.value
    return reward, done


# state的Z-zero归一化
def normalize_state(state):
    norm_state = np.zeros(state_dim)
    for i in range(state_dim):
        if state[i] == -1:  # 缺省值暂时设置为-1, 所以不需要做norm
            norm_state[i] = -1
            continue
        norm_state[i] = (state[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return norm_state


# 打乱用于训练的数据
def shuffle(returns, actions, logprobs, states):
    shuffle_data = list(zip(returns, actions, logprobs, states))
    random.shuffle(shuffle_data)
    new_returns, new_actions, new_logprobs, new_states = zip(*shuffle_data)
    return new_returns, new_actions, new_logprobs, new_states


# 打印所有超参数
def print_config():
    print("============================================================================================")
    if torch.cuda.is_available():
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("training environment name : " + env_name)
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    print("save checkpoint path : " + checkpoint_path)
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", k_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
    print("============================================================================================")


def obs_postprocess(raw_obs, mean_obs, std_obs):
    """trick6, 状态标准化; trick7, observation clipping"""
    """这个不实现, 因为每个环境的状态维度都不一样, 但是原理就是正规化, 即x = (x - x_mean) / x_std"""
    # 这里是数组操作, 因为不同特征的取值范围都不一样, 比如坐标是0-100, 伤害是0-10, 所以坐标和伤害的正规化要各自计算
    # new_obs = (raw_obs - mean_obs) / std_obs
    # 然后有一些数据正规化了之后还是过大过小, 需要再做一波clip
    # new_obs = np.clip(new_obs, -10, 10)
    raise NotImplementedError


# 学习率线性衰减
def update_linear_schedule(optimizer, timesteps, total_timesteps):
    """负责控制actor和critic学习率的线性衰减, 衰减力度都可以通过公式调"""
    ratio = 1 - timesteps / float(total_timesteps)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= ratio


# 开启进程
def env_run():  # 注意！路径写死！
    p = subprocess.Popen(env_path, shell=True, stdout=subprocess.PIPE)
    p.communicate()


# 关闭进程
def env_close(old_env_proc):  # old_client
    subprocess.call("taskkill /F /T /PID {}".format(old_env_proc.pid))
    old_env_proc.terminate()
    # TODO 结束已有的client连接


# 重置进程
def reset(old_env_proc):
    if old_env_proc != '':
        env_close(old_env_proc)
    p = Process(target=env_run, name='env_star')
    p.start()
    return p
