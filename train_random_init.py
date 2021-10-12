from datetime import datetime

from PPO import PPO
from client import MyClient
from common import *

# 设置随机种子
if random_seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def train():
    # 打印参数信息
    print_config()

    env_proc = ""

    # 初始化PPO
    ppo_agent = PPO()
    if load_flag and os.path.exists(checkpoint_path):
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
    done_type = [0] * 7  # TODO, 改成自动获得枚举个数的形式

    client = MyClient()

    # 开始训练
    while time_step <= max_training_timesteps:
        """trick4, 学习率下降而非固定不变"""
        if use_linear_lr_decay:
            update_linear_schedule(ppo_agent.optimizer, time_step, max_training_timesteps)

        # print(datetime.now().replace(microsecond=0) - start_time)
        connect_flag = False  # C++和py是否完成通信连接
        while not connect_flag:
            env_proc = reset(env_proc)  # 双端测试时注释掉
            client.send_reset()
            state, connect_flag = client.poll_reset()
        current_ep_reward = 0
        # print(datetime.now().replace(microsecond=0) - start_time)

        for t in range(max_ep_len):
            time_step += 1
            # 环境交互
            action_send = np.zeros(3)
            action_send[2] = state[2]
            norm_state = normalize_state(state)

            action, logprob, state_value = ppo_agent.select_action(norm_state)
            angle = ((state[5] + action[0] * 30) % 360) / 180.0 * math.pi
            action_send[0], action_send[1] = math.cos(angle), math.sin(angle)
            client.send_action(action_send)
            next_state, reward, done = client.recv_step(state)
            # print(time_step, reward, current_ep_reward, done)
            """trick5, reward clipping"""
            # reward = np.clip(reward, -5, 5)
            current_ep_reward += reward

            # buffer存一帧数据
            ppo_agent.buffer.append(action, logprob, state, next_state, state_value, reward, done)

            # 更新PPO
            if ppo_agent.buffer.is_full() and time_step % update_timestep == 0:
                ppo_agent.update_sgd()

            # 对于连续动作, 隔段时间降低动作标准差, 保证策略收敛
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std()

            # log 记录 average reward
            if time_step % log_freq == 0:

                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, log_running_reward, log_running_episodes))
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

            if t == max_ep_len - 1:
                done = Done.time_out.value
            if done:  # 结束episode
                ppo_agent.writer.add_scalar('reward', current_ep_reward, global_step=i_episode)
                done_type[done] += 1
                break

            state = next_state

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        if i_episode % 100 == 0:
            ppo_agent.writer.add_scalar('metrics/not_done', done_type[0], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/time_out', done_type[1], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/red_crash', done_type[2], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/blue_crash', done_type[3], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/arrive_goal', done_type[4], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/self_crash_angle', done_type[5], global_step=i_episode)
            ppo_agent.writer.add_scalar('metrics/self_crash_coord', done_type[6], global_step=i_episode)
            done_type = [0] * 7

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
