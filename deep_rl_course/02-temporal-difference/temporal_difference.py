import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values
import random


def init():
    env = gym.make('CliffWalking-v0')

    print("-"*60)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("-"*60)
    print()

    # define the optimal state-value function
    V_opt = np.zeros((4,12))
    V_opt[0:13][0] = -np.arange(3, 15)[::-1]
    V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
    V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
    V_opt[3][0] = -13

    plot_values(V_opt)

    return env


def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.
    
    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    # exploitation
    if random.random() > eps:           # select greedy action with probability epsilon
        return np.argmax(Q[state])
    # exploration
    else:                               # otherwise, select an action randomly
        return random.choice(np.arange(nA))
    
def update_Q_sarsa_zero(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    """Returns updated Q-value for the most recent experience."""
    # Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t+1 + gamma * Q(S_t+1, A_t+1) - Q(S_t, A_t))
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)

    # get value of state, action pair at next time step
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0    
    target = reward + (gamma * Qsa_next)               # construct TD target
    new_value = current + (alpha * (target - current)) # get updated value
    
    return new_value

def sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    nA = env.action_space.n                # number of actions
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(nA))

    # initialize performance monitor
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}/{num_episodes}", end="")
            sys.stdout.flush()   
    
        # generate an episode
        score = 0                                             # initialize score
        state, info = env.reset()                             # start episode
        
        # select epsilon-greedy action
        eps = 1.0 / i_episode                                 # set value of epsilon
        action = epsilon_greedy(Q, state, nA, eps)            # epsilon-greedy action selection
        
        while True:
            next_state, reward, done, trunc, info = env.step(action)

            if not done:
                # select epsilon-greedy action
                next_action = epsilon_greedy(Q, next_state, nA, eps) 

                # update Q
                Q[state][action] = update_Q_sarsa_zero(alpha, gamma, Q, state, action, reward, next_state, next_action)

                # update state and action
                state = next_state     # S <- S'
                action = next_action   # A <- A'
            
            score += reward 

            if done:
                Q[state][action] = update_Q_sarsa_zero(alpha, gamma, Q, state, action, reward)
                tmp_scores.append(score)
                break
        
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))
    

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel(f'Episode Number')
    plt.ylabel(f'Average Reward (Over Next {plot_every} Episodes)')
    plt.show()

    # print best 100-episode performance
    print(f'Best Average Reward over {plot_every} Episodes: {np.max(avg_scores)}')    

    return Q



def update_Q_sarsa_max(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    """Returns updated Q-value for the most recent experience."""
    pass

def update_Q_sarsa_expected(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    """Returns updated Q-value for the most recent experience."""
    pass


def run_sarsa_zero():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)
    check(Q_sarsa)



def check(Q_sarsa):
    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


if __name__ == '__main__':
    env = init()
    run_sarsa_zero()