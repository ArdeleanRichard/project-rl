import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym

from params import BASE_RANDOM_SEED


def train_q_learning(
    env,
    use_action_mask: bool = True,
    episodes: int = 5000,
    seed: int = BASE_RANDOM_SEED,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
) -> dict:
    """Train a Q-learning agent with or without action masking."""
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Initialize Q-table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    # Track episode rewards for analysis
    episode_rewards = []

    for episode in range(episodes):
        # Reset environment
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Get action mask if using it
            action_mask = info["action_mask"] if use_action_mask else None

            # Epsilon-greedy action selection with masking
            if np.random.random() < epsilon:
                # Random action selection
                if use_action_mask:
                    # Only select from valid actions
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    # Select from all actions
                    action = np.random.randint(0, n_actions)
            else:
                # Greedy action selection
                if use_action_mask:
                    # Only consider valid actions for exploitation
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[np.argmax(q_table[state, valid_actions])]
                    else:
                        action = np.random.randint(0, n_actions)
                else:
                    # Consider all actions
                    action = np.argmax(q_table[state])

            # Take action and observe result
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Q-learning update
            if not (done or truncated):
                if use_action_mask:
                    # Only consider valid next actions for bootstrapping
                    next_mask = info["action_mask"]
                    valid_next_actions = np.nonzero(next_mask == 1)[0]
                    if len(valid_next_actions) > 0:
                        next_max = np.max(q_table[next_state, valid_next_actions])
                    else:
                        next_max = 0
                else:
                    # Consider all next actions
                    next_max = np.max(q_table[next_state])

                # Update Q-value
                q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_factor * next_max - q_table[state, action]
                )

            state = next_state

        episode_rewards.append(total_reward)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }
