from collections import defaultdict

import numpy as np
import gymnasium as gym

from utils.utils import get_space_size


class BaseAgent:
    def __init__(self, env, config):
        """ Initialize agent.

        Params
        ======
        """

        self.config = config
        self.env = env

        self.n_actions = get_space_size(self.env.action_space)
        self.n_states = get_space_size(self.env.observation_space)

        self.lr =                   self.config["learning_rate"]
        self.discount_factor =      self.config["discount_factor"]  # How much we care about future rewards

        # Exploration parameters
        self.epsilon =              self.config["start_epsilon"]
        self.epsilon_decay =        self.config["epsilon_decay"]
        self.final_epsilon =        self.config["final_epsilon"]

        self.name = config["name"]
        if "name_ext" in config:
            self.name_ext = config["name_ext"]

        # Track learning progress
        self.training_error = []


    def select_action(self, state, info):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        pass

    def update(self, state, info, action, reward, done, next_state):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        pass

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class BaseSarsaAgent(BaseAgent):
    def __init__(self, env: gym.Env, config: dict = None):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        super().__init__(env, config)

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))
        # self.q_values = np.zeros((self.n_states, self.n_actions))

        self.use_action_mask =      self.config["use_action_mask"]



    def reset_qtable(self):
        """Reset the Q-table."""
        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))
        # self.q_values = np.zeros((self.n_states, self.n_actions))


    def epsilon_greedy(self, state, info):
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[state]))


    def select_action(self, state, info):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        pass

    def update(self, state, info, action, reward, done, next_state):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        pass