import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2, device='cpu'):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)



def reinforce(env, policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _,  _ = env.step(action)

            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')
        if np.mean(scores_deque) >= 195.0:
            print(f'Environment solved in {i_episode:d} episodes!\n\tAverage Score: {np.mean(scores_deque):.2f}')
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
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _, _ = env.step(action)
        if done:
            break

    env.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')
    obs, info = env.reset(seed=0)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    policy = Policy(device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    scores = reinforce(env, policy, optimizer)

    plot_scores(scores)
    watch_agent(policy)
