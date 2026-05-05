# use action masking in the Taxi environment to improve reinforcement learning performance by preventing invalid actions.
# Action masking is a technique that helps reinforcement learning agents avoid selecting invalid actions by
# providing a binary mask that indicates which actions are valid in the current state.
# - This can significantly improve learning efficiency and performance.

# How Action Masking Works
# Action masking works by constraining the agent’s action selection to only valid actions:
# - During exploration: When selecting random actions, we only choose from the set of valid actions
# - During exploitation: When selecting the best action based on Q-values, we only consider Q-values for valid actions
# - During Q-learning updates: We compute the maximum future Q-value only over valid actions in the next state

# Key Benefits of Action Masking:
# 1. Faster Convergence: Agents with action masking typically learn faster because they don’t waste time exploring invalid actions.
# 2. Better Performance: By focusing only on valid actions, the agent can achieve higher rewards more consistently.
# 3. More Stable Learning: Action masking reduces the variance in learning by eliminating the randomness associated with invalid action selection.
# 4. Practical Applicability: In real-world scenarios, action masking prevents agents from taking actions that could be dangerous or impossible.
#
# Reminder of Key Implementation Details
# - Action Selection: We filter available actions using np.nonzero(action_mask == 1)[0] to get only valid actions
# - Q-Value Updates: When computing the maximum future Q-value, we only consider valid actions in the next state
# - Exploration: Random action selection is constrained to the set of valid actions

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym

from q_learning import train_q_learning
from params import BASE_RANDOM_SEED, n_runs, learning_rate, discount_factor, epsilon, episodes


def runs():
    # Generate different seeds for each run
    seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]

    # Store results for comparison
    masked_results_list = []
    unmasked_results_list = []

    # Run experiments with different seeds
    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{n_runs} with seed {seed}")

        # Train agent WITH action masking
        env_masked = gym.make("Taxi-v4")
        masked_results = train_q_learning(
            env_masked,
            use_action_mask=True,
            seed=seed,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            episodes=episodes,
        )
        env_masked.close()
        masked_results_list.append(masked_results)

        # Train agent WITHOUT action masking
        env_unmasked = gym.make("Taxi-v4")
        unmasked_results = train_q_learning(
            env_unmasked,
            use_action_mask=False,
            seed=seed,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            episodes=episodes,
        )
        env_unmasked.close()
        unmasked_results_list.append(unmasked_results)

    return masked_results_list, unmasked_results_list

def plot(masked_results_list, unmasked_results_list):
    # Calculate statistics across runs
    masked_mean_rewards = [r["mean_reward"] for r in masked_results_list]
    unmasked_mean_rewards = [r["mean_reward"] for r in unmasked_results_list]

    masked_overall_mean = np.mean(masked_mean_rewards)
    masked_overall_std = np.std(masked_mean_rewards)
    unmasked_overall_mean = np.mean(unmasked_mean_rewards)
    unmasked_overall_std = np.std(unmasked_mean_rewards)

    # Create visualization
    plt.figure(figsize=(12, 8), dpi=100)

    # Plot individual runs with low alpha
    for i, (masked_results, unmasked_results) in enumerate(
            zip(masked_results_list, unmasked_results_list, strict=True)
    ):
        plt.plot(
            masked_results["episode_rewards"],
            label="With Action Masking" if i == 0 else None,
            color="blue",
            alpha=0.1,
        )
        plt.plot(
            unmasked_results["episode_rewards"],
            label="Without Action Masking" if i == 0 else None,
            color="red",
            alpha=0.1,
        )

    # Calculate and plot mean curves across all runs
    masked_mean_curve = np.mean([r["episode_rewards"] for r in masked_results_list], axis=0)
    unmasked_mean_curve = np.mean(
        [r["episode_rewards"] for r in unmasked_results_list], axis=0
    )

    plt.plot(
        masked_mean_curve, label="With Action Masking (Mean)", color="blue", linewidth=2
    )
    plt.plot(
        unmasked_mean_curve,
        label="Without Action Masking (Mean)",
        color="red",
        linewidth=2,
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance: Q-Learning with vs without Action Masking")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    savefig_folder = Path("_static/img/tutorials/")
    savefig_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        savefig_folder / "taxi_v3_action_masking_comparison.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.savefig("action_masked_vs_unmasked.png")
    plt.close()


if __name__ == "__main__":
    masked_results_list, unmasked_results_list = runs()
    plot(masked_results_list, unmasked_results_list)