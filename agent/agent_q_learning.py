import gymnasium as gym
import numpy as np

from agent.agent_base import BaseSarsaAgent


class AgentQLearning(BaseSarsaAgent):
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

    def epsilon_greedy(self, state, info):
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        action_mask = info["action_mask"] if self.config["use_action_mask"] else None

        if self.config["use_action_mask"] == False:
            # With probability epsilon: explore (random action)
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

            # With probability (1-epsilon): exploit (best known action)
            else:
                return int(np.argmax(self.q_values[state]))
        elif self.config["use_action_mask"] == True:
            if np.random.random() < self.epsilon:
                # Only select from valid actions
                valid_actions = np.nonzero(action_mask == 1)[0]
                action = np.random.choice(valid_actions)
                return action

            else:
                # Only consider valid actions for exploitation
                valid_actions = np.nonzero(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[np.argmax(self.q_values[state][valid_actions])]
                else:
                    action = np.random.randint(0, self.n_actions)
                return action

    def select_action(self, state: tuple[int, int, bool], info) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        return self.epsilon_greedy(state, info)

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
        if self.config["use_action_mask"] == True:
            # Only consider valid next actions for bootstrapping
            next_mask = info["action_mask"]
            valid_next_actions = np.nonzero(next_mask == 1)[0]
            if len(valid_next_actions) > 0:
                next_max = np.max(self.q_values[next_state][valid_next_actions])
            else:
                next_max = 0
            future_q_value = (not terminated) * next_max
        elif self.config["use_action_mask"] == False:
            # Consider all next actions
            future_q_value = (not terminated) * np.max(self.q_values[next_state])


        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (self.q_values[state][action] + self.lr * temporal_difference)

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

