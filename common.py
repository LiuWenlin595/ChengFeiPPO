import math
import numpy as np
from multiprocessing import Process
import subprocess

from config import *


# 计算reward和done
def get_reward_done(cur_state, next_state, red_crash, blue_crash):
    reward, done = 0, False
    # done的情况: 红方被击毁, 蓝方被击毁, 红方到达目标点, 红方坠机, 到达最长时间

    # 根据目标点距离给予连续性小奖励
    goal_lat, goal_lon, goal_height = cur_state[11], cur_state[12], cur_state[13]
    cur_dist = math.sqrt(pow(cur_state[0] - goal_lat, 2) + pow(cur_state[1] - goal_lon, 2))
    next_dist = math.sqrt(pow(next_state[0] - goal_lat, 2) + pow(next_state[1] - goal_lon, 2))
    reward += (cur_dist - next_dist) * 1000  # 乘以一个数量级, 防止dist过小

    # TODO 根据导弹的范围来设计奖励, 但是导弹并不是全局信息, 所以需要考虑疏忽的情况

    # 到达目标点给予一次性大奖励, 如果直接朝目标点飞的话大概一次移动0.00027, 4000步可以走1.08没问题
    if cur_dist < 0.1:
        print("arrive goal!")
        reward += 5
        done = True

    # 红方被击中, 给予一次性大惩罚
    if red_crash:
        print("red crash!")
        reward -= 5
        done = True

    # 蓝方被击中, 给予一次性大奖励
    if blue_crash:
        print("blue crash!")
        reward += 5

    # 坠机, 给予一次性大惩罚
    delta_theta = abs(next_state[4] - cur_state[4])
    if delta_theta > 180:
        delta_theta = 360 - delta_theta
    delta_psi = abs(next_state[5] - cur_state[5])
    if delta_psi > 180:
        delta_psi = 360 - delta_psi
    if delta_theta > 15 or delta_psi > 15:
        print("crash down! ", delta_theta, delta_psi, next_state[4], cur_state[4], next_state[5], cur_state[5])
        reward -= 5
        done = True
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
