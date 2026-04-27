import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.005
        self.alpha = 0.10
        self.gamma = 1.0


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # ### Epsilon-Greedy

        # exploitation - select greedy action with probability epsilon
        if random.random() > self.eps: 
            return np.argmax(self.Q[state])
        
        # exploration - otherwise, select an action randomly
        else: 
            return random.choice(np.arange(self.nA))
        
        # def epsilon_greedy_probs(self, Q_s):
        #     # self.eps =  max(self.eps * 0.99999, 0.0005)
        #     policy_s = np.ones(self.nA) * self.eps / self.nA
        #     policy_s[np.argmax(Q_s)] = 1 - self.eps + (self.eps / self.nA)
        #     return policy_s


        # return np.random.choice(np.arange(self.nA), p=self.epsilon_greedy_probs(self.Q[state]))



    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1

        ### Q Learning (SarsaMax)
        # current = self.Q[state][action]
        # Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0   # value of next state 
        # target = reward + gamma * Qsa_next
        # self.Q[state][action] = current + alpha * (target - current)


        ### ExpectedSarsa (no episode number known - this is probably better)
        current = self.Q[state][action]

        policy_s = np.ones(self.nA) * self.eps / self.nA                                    # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)       # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s)                                     # get value of state at next time step

        target = reward + self.gamma * Qsa_next
        self.Q[state][action] = current + self.alpha * (target - current)

        