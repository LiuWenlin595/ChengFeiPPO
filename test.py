import time
import gym

from PPO import PPO
from config import *


def test():
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim)

    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(total_test_episodes):
        ep_reward = 0
        state = env.reset()

        for t in range(max_ep_len):

            action, _, _ = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


if __name__ == '__main__':
    test()
