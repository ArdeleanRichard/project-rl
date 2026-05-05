import gymnasium as gym
import ale_py

import time
import matplotlib
import matplotlib.pyplot as plt
import torch

import torch.optim as optim

# custom utilies for displaying animation, collecting rollouts and more
from utils import pong_utils
from utils.parallelEnv import parallelEnv
from reinforce_pong import Policy, surrogate, train


def read_frame(env):
    # show what a preprocessed image looks like
    env.reset()
    _, _, _, _, _ = env.step(0)

    # get a frame after 20 steps
    for _ in range(20):
        frame, _, _, _, _ = env.step(1)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1, 2, 2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
    plt.savefig('pong_preprocessed_image.png')
    plt.close()


def create_env():
    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", frameskip=4, repeat_action_probability=0.0)
    # env = gym.make('PongDeterministic-v4')
    print("List of available actions: ", env.unwrapped.get_action_meanings())
    return env


def run_reinforce():
    # check which device is being used.
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ", device)

    env = create_env()

    read_frame(env)


    # Get policy
    # policy = Policy().to(device)
    policy = pong_utils.Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)


    # try to add the option "preprocess=pong_utils.preprocess_single"
    # to see what the agent sees
    pong_utils.play(env, policy, time=100)

    # envs = parallelEnv('PongDeterministic-v4', n=4, seed=12345)
    envs = parallelEnv('ALE/Pong-v5', n=4, seed=12345)
    prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)
    print(reward)

    # Lsur = surrogate(policy, prob, state, action, reward)
    # print(Lsur)

    mean_rewards = train(policy, optimizer)

    # play game after training!
    pong_utils.play(env, policy, time=2000)

    plt.plot(mean_rewards)
    plt.show()

    # save your policy
    torch.save(policy, 'REINFORCE.policy')


def run_ppo():
    # check which device is being used.
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ", device)

    env = create_env()

    read_frame(env)

    # Get policy
    # policy = Policy().to(device)
    policy = pong_utils.Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # try to add the option "preprocess=pong_utils.preprocess_single"
    # to see what the agent sees
    pong_utils.play(env, policy, time=100)



if __name__ == "__main__":
    run_reinforce()
    run_ppo()
