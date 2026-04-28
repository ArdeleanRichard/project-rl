import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import AgentDQN, AgentDoubleDQN, AgentPriorityDQN, AgentDuelingDQN, AgentDistributionalDQN, AgentNoisyDQN, AgentRainbow



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
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
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
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _, _ = env.step(action)
        if done:
            break 
            
    env.close()


def test_agent(env, agent, savefile):
    env = gym.make('LunarLander-v3', render_mode='human')
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(savefile))

    for i in range(5):
        watch_agent(env, agent)
                
    env.close()






def run_dqn():
    env = prep_env()
    agent = AgentDQN(state_size=8, action_size=4, seed=0)
    # watch_agent(env, agent)

    savefile = "dqn_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(env, agent, savefile=savefile)



def run_dqn_double():
    env = prep_env()
    agent = AgentDoubleDQN(state_size=8, action_size=4, seed=0)

    savefile = "dqn_double_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(env, agent, savefile=savefile)


def run_dqn_priority():
    env = prep_env()
    agent = AgentPriorityDQN(state_size=8, action_size=4, seed=0)

    savefile = "dqn_priority_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(env, agent, savefile=savefile)


def run_dqn_dueling():
    env = prep_env()
    agent = AgentDuelingDQN(state_size=8, action_size=4, seed=0)

    savefile = "dqn_dueling_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(env, agent, savefile=savefile)


def run_dqn_distributional():
    """
    C51: predicts a probability distribution over returns instead of a scalar.
    Selects actions by argmax of expected Q = Σ z_i · p_i.
    Uses ε-greedy (like standard DQN) since there is no built-in noise.
    """
    env    = prep_env()
    agent  = AgentDistributionalDQN(state_size=8, action_size=4, seed=0)
    
    savefile = "dqn_distributional_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)
    test_agent(env, agent, savefile)
 
 
def run_dqn_noisy():
    """
    Noisy DQN: NoisyLinear layers provide intrinsic, state-dependent exploration.
    No ε schedule is needed — pass eps_start=0 so the loop never applies ε-greedy.
    """
    env   = prep_env()
    agent = AgentNoisyDQN(state_size=8, action_size=4, seed=0)

    savefile = "dqn_noisy_checkpoint.pth"
    # eps_start=0: the noisy network handles all exploration internally
    scores = dqn(env, agent, savefile=savefile, eps_start=0.0, eps_end=0.0, eps_decay=1.0)
    plot_scores(scores)
    test_agent(env, agent, savefile)
 
 
def run_rainbow():
    """
    Rainbow: Double + Prioritized + Dueling + Multi-step + Distributional + Noisy.
    No ε schedule — NoisyLinear handles exploration.
    """
    env   = prep_env()
    agent = AgentRainbow(state_size=8, action_size=4, seed=0, n_steps=3)

    
    savefile = "dqn_rainbow_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile, eps_start=0.0, eps_end=0.0, eps_decay=1.0)
    plot_scores(scores)
    test_agent(env, agent, savefile)
 


if __name__ == "__main__":
    # run_dqn()
    # run_dqn_double()
    # run_dqn_priority()

    # Not working 
    # run_dqn_dueling()
    # run_dqn_distributional()
    # run_dqn_noisy()
    run_rainbow()

    pass


