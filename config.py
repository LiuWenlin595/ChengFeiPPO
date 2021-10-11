import os
import torch
"""-------------------------配置超参数-------------------------"""
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
"""-------------------------环境超参数-------------------------"""
env_name = "1v1"
env_path = "D:\\Project\\skln_working\\dev_asnb_1v1_r4.bat"
state_dim = 25  # 状态空间维度
min_max = [[-1] * 2 for _ in range(state_dim)]  # 用来存所有state的min值和max值
min_max[0] = min_max[11] = min_max[14] = [36.5, 37.5]
min_max[1] = min_max[12] = min_max[15] = [-122, -121]
min_max[2] = min_max[13] = min_max[16] = [5000, 6000]
min_max[3] = min_max[4] = min_max[5] = min_max[17] = min_max[18] = min_max[19] = [0, 360]
min_max[6] = min_max[7] = min_max[8] = min_max[20] = min_max[21] = min_max[22] = min_max[24] = [0, 250]
min_max[9] = min_max[10] = [0, 1]
min_max[23] = [0, 5000]
action_dim = 1  # 动作空间维度
has_continuous_action_space = True  # 连续动作为True否则False

max_ep_len = 4000  # 一个episode的最多timesteps
max_training_timesteps = int(3e7)  # 当 timesteps > max_training_timesteps 时停止循环

print_freq = max_ep_len * 10  # 每隔多少个step打印一次 average reward
log_freq = max_ep_len * 10  # 每隔多少个step将 average reward 保存到日志
save_model_freq = print_freq * 10  # 每隔多少个step保存一次模型
load_flag = True  # 是否接着上一次的checkpoint继续训练
"""-------------------------模型超参数-------------------------"""
action_std = 0.3  # Multivariate Normal动作分布的初始标准差
action_std_decay_rate = 0.05  # 标准差的线性衰减步长 (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # 最小动作标准差, 当 action_std <= min_action_std 时停止衰减
action_std_decay_freq = int(1e6)  # 每隔 action_std_decay_freq 衰减一次

lr_actor = 0.0003  # actor学习率
lr_critic = 0.001  # critic学习率
use_orth = False  # 是否使用正交初始化参数
use_linear_lr_decay = False  # 是否使用学习率衰减

critic_coef = 0.5  # critic loss 权重
entropy_coef = 0.01  # entropy loss 权重
use_value_clip = False  # 是否使用critic value clip
max_grad_norm = 1  # 梯度clip的约束范围

random_seed = 1  # 设定随机种子, 0表示不设随机种子
run_num_pretrained = 0  # 更改这个值来保证新保持的model不会覆盖之前的

log_dir = "logs/" + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

directory = "models/" + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
tensorboard_path = directory
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
"""-------------------------训练超参数-------------------------"""
update_timestep = max_ep_len * 3  # 每隔 update_timestep 执行一次 update policy
k_epochs = 3  # 一个 update policy 中更新k轮, 相当于每个traj使用k_epochs遍

eps_clip = 0.2  # clip参数
gamma = 0.99  # 折扣因子

use_gae = True  # 是否使用GAE
gae_lambda = 0.95  # gae的权重参数
mini_batch = 12000  # 单批数据的处理量
batch_size = 30000  # mempool的容量
"""-------------------------测试超参数-------------------------"""
render = True  # 是否render
frame_delay = 0  # 是否每一帧停顿一些时间, 可以render的更清楚
total_test_episodes = 10  # 测试轮次
"""-----------------------------------------------------------"""
