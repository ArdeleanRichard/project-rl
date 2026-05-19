"""
Policy-based methods are a class of algorithms that search directly for the optimal policy WITHOUT simultaneously maintaining value function estimates
- Policy gradient Methods - estimate weights of an optimal policy through gradient ascent
"""

import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        
    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))
    
    def act(self, state):
        probs = self.forward(state)
        #action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)              # option 2: deterministic policy
        return action
    


def hill_climbing(env, policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.
        
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.inf
    best_w = policy.w

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        # found better weights
        if R >= best_R: 
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape) 
        # did not find better weights
        else: 
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')
        if np.mean(scores_deque)>=195.0:
            print(f'Environment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
            policy.w = best_w
            break
        
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def watch_agent(policy):
    # test agent
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()

    for t in range(200):
        action = policy.act(state)
        env.render()
        state, reward, done, _, _ = env.step(action)
        if done:
            break

    env.close()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    obs, info = env.reset(seed=0)
    np.random.seed(0)

    policy = Policy()

    scores = hill_climbing(env, policy)

    plot_scores(scores)
    watch_agent(policy)

