import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



def prep_env():
    # Prepare environment
    env = gym.make('LunarLander-v3')
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    return env


def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, savefile="checkpoint.pth"):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window)>=200.0:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\n\tAverage Score: {np.mean(scores_window):.2f}')
            # print(f'\nEnvironment solved in {i_episode-100:d} episodes!\n\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), savefile)
            break
    
    return scores


def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()



def watch_agent(env, agent):
    # watch an untrained agent
    state, _ = env.reset()

    total_rewards_ep = 0
    done = False
    while not done:
        action = agent.act(state)

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_rewards_ep += reward

    return total_rewards_ep


def test_agent(agent, savefile, n_eval_episodes=5):
    env = gym.make('LunarLander-v3', render_mode='human')
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(savefile))

    episode_rewards = []
    # for episode in tqdm(range(n_eval_episodes)):
    for i in range(n_eval_episodes):
        total_reward = watch_agent(env, agent)

        episode_rewards.append(total_reward)
                
    env.close()

    mean_reward = np.mean(np.array(episode_rewards))
    std_reward = np.std(np.array(episode_rewards))

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")


