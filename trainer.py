import os
from tqdm import tqdm  # Progress bar

import numpy as np
from collections import defaultdict, deque

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from environment_creator import EnvironmentCreator


class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train_agent(self):
        # Track episode rewards for analysis
        all_rewards = []
        rewards_window = deque(maxlen=100)

        pbar = tqdm(range(self.env.config['n_episodes']), desc="Training")

        for episode in pbar:
            # Start a new hand
            state, info = self.env.reset()
            done = False
            total_reward = 0

            # Play one episode
            t = 0
            while not done:
                # Agent chooses action (initially random, gradually more intelligent)
                action = self.agent.select_action(state, info)

                # Take action and observe result
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Learn from this experience
                self.agent.update(state, info, action, reward, terminated, next_state)

                # Move to next state
                done = terminated or truncated
                state = next_state
                total_reward += reward

                if "max_t" in self.agent.config:
                    if t >= self.agent.config["max_t"]:
                        break
                t+=1

            # Reduce exploration rate (agent becomes less random over time)
            self.agent.decay_epsilon()

            all_rewards.append(total_reward)
            rewards_window.append(total_reward)

            pbar.set_postfix({
                "eps": f"{self.agent.epsilon:.3f}",
                "reward": f"{total_reward:.2f}",
                "avg100": f"{np.mean(rewards_window):.2f}"
            })

            if episode % (self.env.config['n_episodes'] // 10) == 0:
                print(f'\rEpisode {episode}\t\tEps: {self.agent.epsilon:.3f}\t\tAverage Score: {np.mean(rewards_window):.2f}')

            if "score_threshold" in self.env.config:
                if np.mean(rewards_window) >= self.env.config["score_threshold"]:
                    print(f'\nEnvironment solved in {episode:d} episodes!\n\tAverage Score: {np.mean(rewards_window):.2f}')
                    break

        return all_rewards


    # Test the trained agent
    def test_agent(self):
        """Test agent performance without learning or exploration."""
        env_test = EnvironmentCreator.create_test_env(self.env.config)
        all_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # Pure exploitation

        # for _ in range(self.env.config["test_n_episodes"]):
        for _ in tqdm(range(self.env.config['test_n_episodes']), desc="Testing"):
            state, info = env_test.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, info)
                state, reward, terminated, truncated, info = env_test.step(action)
                episode_reward += reward
                done = terminated or truncated

            all_rewards.append(episode_reward)

        # Restore original epsilon
        self.agent.epsilon = old_epsilon

        env_test.close()

        win_rate = np.mean(np.array(all_rewards) > 0)

        print(f"Test Results over {self.env.config["test_n_episodes"]} test episodes:")
        print(f"\tWin Rate: {win_rate:.1%}")
        print(f"\tAverage Reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")

    def save_agent(self):
        self.agent.save_checkpoint()

    # -----------------------------------------------------------------
    # Visualize Training
    # -----------------------------------------------------------------
    def get_moving_avgs(self, arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    def plot_train(self, savefolder="./plots/"):
        os.makedirs(os.path.dirname(savefolder), exist_ok=True)

        # Smooth over a 500-episode window
        rolling_length = 500
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = self.get_moving_avgs(
            self.env.return_queue,
            rolling_length,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = self.get_moving_avgs(
            self.env.length_queue,
            rolling_length,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = self.get_moving_avgs(
            self.agent.training_error,
            rolling_length,
            "same"
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        save_file = savefolder+f"agent_{self.agent.name}.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Plot saved to {save_file}")

    # -----------------------------------------------------------------
    # Visualize Policy
    # -----------------------------------------------------------------
    def plot_policy(self, savefolder):
        if self.env.config['name'] == "Blackjack-v1":
            viz = BlackjackVisualizer(self.agent)
            viz.plot_policy(savefolder=savefolder)

        if self.env.config['name'] == "FrozenLake-v1":
            viz = FrozenlakeVisualizer(self.env, self.agent, self.env.config)
            viz.plot_q_values_map(savefolder=savefolder)


class BlackjackVisualizer:
    def __init__(self, agent):
        self.agent = agent

    # -----------------------------------------------------------------
    # Visualize the policy
    # -----------------------------------------------------------------
    def create_grids(self, usable_ace=False):
        """Create value and policy grid given an agent."""
        # convert our state-action values to state values
        # and build a policy dictionary that maps observations to actions
        state_value = defaultdict(float)
        policy = defaultdict(int)
        for obs, action_values in self.agent.q_values.items():
            state_value[obs] = float(np.max(action_values))
            policy[obs] = int(np.argmax(action_values))

        player_count, dealer_count = np.meshgrid(
            # players count, dealers face-up card
            np.arange(12, 22),
            np.arange(1, 11),
        )

        # create the value grid for plotting
        value = np.apply_along_axis(
            lambda obs: state_value[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        value_grid = player_count, dealer_count, value

        # create the policy grid for plotting
        policy_grid = np.apply_along_axis(
            lambda obs: policy[(obs[0], obs[1], usable_ace)],
            axis=2,
            arr=np.dstack([player_count, dealer_count]),
        )
        return value_grid, policy_grid

    def create_plots(self, value_grid, policy_grid, title: str):
        """Creates a plot using a value and policy grid."""
        # create a new figure with 2 subplots (left: state values, right: policy)
        player_count, dealer_count, value = value_grid
        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(title, fontsize=16)

        # plot the state values
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            player_count,
            dealer_count,
            value,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        plt.xticks(range(12, 22), range(12, 22))
        plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
        ax1.set_title(f"State values: {title}")
        ax1.set_xlabel("Player sum")
        ax1.set_ylabel("Dealer showing")
        ax1.zaxis.set_rotate_label(False)
        ax1.set_zlabel("Value", fontsize=14, rotation=90)
        ax1.view_init(20, 220)

        # plot the policy
        fig.add_subplot(1, 2, 2)
        ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
        ax2.set_title(f"Policy: {title}")
        ax2.set_xlabel("Player sum")
        ax2.set_ylabel("Dealer showing")
        ax2.set_xticklabels(range(12, 22))
        ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

        # add a legend
        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
            Patch(facecolor="grey", edgecolor="black", label="Stick"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
        return fig

    def plot_policy(self, savefolder="./plots/policy"):
        # state values & policy with usable ace (ace counts as 11)
        value_grid, policy_grid = self.create_grids(usable_ace=True)
        fig1 = self.create_plots(value_grid, policy_grid, title="With usable ace")
        plt.savefig(savefolder + f"agent_{self.agent.name}_policy_usable_ace.png")

        # state values & policy without usable ace (ace counts as 1)
        value_grid, policy_grid = self.create_grids(usable_ace=False)
        fig2 = self.create_plots(value_grid, policy_grid, title="Without usable ace")
        plt.savefig(savefolder + f"agent_{self.agent.name}_policy_no_usable_ace.png")



class FrozenlakeVisualizer:
    def __init__(self, env, agent, config_env):
        self.env = env
        self.agent = agent
        self.env.config = config_env

    def qtable_directions_map(self):
        """Get the best learned action & map it to arrows."""
        map_size = self.env.config["map_size"]

        ### These work only if we have numpy matrix - like frozenlake
        # qtable = self.agent.q_values
        # qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
        # qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)

        qtable_val_max = np.zeros((map_size, map_size))
        qtable_best_action = np.zeros((map_size, map_size), dtype=int)

        for state, action_values in self.agent.q_values.items():
            row, col = divmod(state, map_size)

            qtable_val_max[row, col] = np.max(action_values)
            qtable_best_action[row, col] = np.argmax(action_values)

        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
        eps = np.finfo(float).eps  # Minimum float number on the machine
        for idx, val in enumerate(qtable_best_action.flatten()):
            if qtable_val_max.flatten()[idx] > eps:
                # Assign an arrow only if a minimal Q-value has been learned as best action
                # otherwise since 0 is a direction, it also gets mapped on the tiles where
                # it didn't actually learn anything
                qtable_directions[idx] = directions[val]
        qtable_directions = qtable_directions.reshape(map_size, map_size)
        return qtable_val_max, qtable_directions

    def plot_q_values_map(self, savefolder="./plots/"):
        """Plot the last frame of the simulation and the policy learned."""
        qtable_val_max, qtable_directions = self.qtable_directions_map()

        # Plot the last frame
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        ax[0].imshow(self.env.render())
        ax[0].axis("off")
        ax[0].set_title("Last frame")

        # Plot the policy
        sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            ax=ax[1],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title="Learned Q-values\nArrows represent best action")
        for _, spine in ax[1].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")
        plt.savefig(savefolder + f"agent_{self.agent.name}_policy.png")