from agent.agent_base import BaseAgent, BaseSarsaAgent

from collections import defaultdict
import gymnasium as gym
import numpy as np


class AgentSarsaExpected(BaseSarsaAgent):
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

    def select_action(self, state: tuple[int, int, bool], info) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        return self.epsilon_greedy(state)

    def update(
            self,
            state: tuple[int, int, bool],
            info,
            action: int,
            reward: float,
            terminated: bool,
            next_state: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)

        policy_s = np.ones(self.n_actions) * self.epsilon / self.n_actions                                          # current policy (for next state S')
        policy_s[np.argmax(self.q_values[next_state])] = 1 - self.epsilon + (self.epsilon / self.n_actions)         # greedy action
        future_q_value = (not terminated) * np.dot(self.q_values[next_state], policy_s)

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (self.q_values[state][action] + self.lr * temporal_difference)

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

