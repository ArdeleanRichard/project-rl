# import numpy as np
#
# from matplotlib import pyplot as plt
#
# from agent_creator import AgentCreator
# from environment_creator import EnvironmentCreator
# from trainer import Trainer
#
#
# class ExperimentCreator:
#     def __init__(self, savefolder, savefile, configs_env, configs_agent):
#         self.savefolder = savefolder
#         self.savefile = savefile
#         self.configs_env = configs_env
#         self.configs_agent = configs_agent
#
#         # single agent config
#         if isinstance(configs_agent, dict):
#             self.trainers = []
#
#             for config_env in configs_env:
#                 env_creator = EnvironmentCreator(config_env)
#                 env = env_creator.create()
#                 # env_creator.print_info()
#
#                 # Create agent
#                 agent = AgentCreator(env=env, config=configs_agent).create()
#
#                 self.trainers.append(Trainer(env=env, agent=agent, config_env=config_env))
#
#
#
#     def start_experiment(self):
#         # single agent config
#         if isinstance(self.configs_agent, dict):
#             self.results = []
#             for trainer in self.trainers:
#                 result = trainer.train_agent()
#                 self.results.append(result)
#
#
#     def plot(self):
#         if isinstance(self.configs_agent, dict):
#             ### Calculate statistics across runs
#             # mean_rewards = [np.mean(r) for r in self.results]
#             #
#             # overall_mean = np.mean(mean_rewards)
#             # overall_std = np.std(mean_rewards)
#
#             # Create visualization
#             plt.figure(figsize=(12, 8), dpi=100)
#
#             # Plot individual runs with low alpha
#             for i, reward in enumerate(self.results):
#                 plt.plot(reward, label="(Runs)" if i == 0 else None, color="blue", alpha=0.1)
#
#             # Calculate and plot mean curves across all runs
#             mean_curve = np.mean(self.results, axis=0)
#
#             plt.plot(mean_curve, label="(Mean)", color="blue", linewidth=2)
#
#             plt.xlabel("Episode")
#             plt.ylabel("Total Reward")
#             plt.title(f"Training Performance: {self.configs_agent["name"]}")
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#
#             # Save the figure
#             plt.savefig(f"{self.savefolder}/{self.savefile}.png")
#             plt.close()


import numpy as np
from matplotlib import pyplot as plt

from agent_creator import AgentCreator
from environment_creator import EnvironmentCreator
from trainer import Trainer


class ExperimentCreator:
    def __init__(self, savefolder, savefile, configs_env, configs_agent):
        self.savefolder = savefolder
        self.savefile = savefile
        self.configs_env = configs_env
        self.configs_agent = configs_agent

        # Normalize agent configs to list for unified handling
        self.agent_configs_list = self._normalize_agent_configs(configs_agent)

        # Structure: {agent_name: [trainer1, trainer2, ...]}
        self.trainers_by_agent = self._initialize_trainers()

        # Structure: {agent_name: [results]}
        self.results_by_agent = {}

    def _normalize_agent_configs(self, configs_agent):
        """Convert agent configs to list format for unified processing."""
        if isinstance(configs_agent, dict):
            return [configs_agent]
        elif isinstance(configs_agent, list):
            return configs_agent
        else:
            raise ValueError("configs_agent must be dict or list of dicts")

    def _initialize_trainers(self):
        """Create trainers for all agent-environment combinations."""
        trainers_by_agent = {}

        for config_agent in self.agent_configs_list:
            agent_name = config_agent["name"]
            trainers_by_agent[agent_name] = []

            print("-" * 60)
            print(f"Agent: {agent_name}")
            print("-" * 60)
            for config_env in self.configs_env:
                # Create environment
                env_creator = EnvironmentCreator(config_env)
                env = env_creator.create()

                # Create agent
                agent = AgentCreator(env=env, config=config_agent).create()

                # Create and store trainer
                trainer = Trainer(env=env, agent=agent, config_env=config_env)
                trainers_by_agent[agent_name].append(trainer)

        return trainers_by_agent

    def start_experiment(self):
        """Run training for all agents across all environments."""
        self.results_by_agent = {}

        for agent_name, trainers in self.trainers_by_agent.items():
            self.results_by_agent[agent_name] = []

            for trainer in trainers:
                result = trainer.train_agent()
                self.results_by_agent[agent_name].append(result)

    def plot(self):
        """Generate plots for all agents."""
        num_agents = len(self.agent_configs_list)

        if num_agents == 1:
            self._plot_single_agent()
        else:
            self._plot_multiple_agents()

    def _plot_single_agent(self):
        """Plot results for a single agent."""
        agent_name = self.agent_configs_list[0]["name"]
        results = self.results_by_agent[agent_name]

        plt.figure(figsize=(12, 8), dpi=100)

        # Plot individual runs
        for i, reward in enumerate(results):
            plt.plot(reward, label="(Runs)" if i == 0 else None, color="blue", alpha=0.1)

        # Plot mean curve
        mean_curve = np.mean(results, axis=0)
        plt.plot(mean_curve, label="(Mean)", color="blue", linewidth=2)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Performance: {agent_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{self.savefolder}/{self.savefile}.png")
        plt.close()

    def _plot_multiple_agents(self):
        """Plot comparison of multiple agents."""
        # Define color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.agent_configs_list)))

        plt.figure(figsize=(14, 8), dpi=100)

        for idx, (agent_name, results) in enumerate(self.results_by_agent.items()):
            color = colors[idx]

            # Plot individual runs with low alpha
            for i, reward in enumerate(results):
                plt.plot(reward, color=color, alpha=0.1, label=f"{agent_name} (Runs)" if i == 0 else None)

            # Plot mean curve
            mean_curve = np.mean(results, axis=0)
            plt.plot(mean_curve, color=color, linewidth=2.5, label=f"{agent_name} (Mean)")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Performance: Agent Comparison")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{self.savefolder}/{self.savefile}.png")
        plt.close()

    def get_statistics(self):
        """Calculate and return statistics for all agents."""
        stats = {}

        for agent_name, results in self.results_by_agent.items():
            results_array = np.array(results)
            mean_rewards = np.mean(results_array, axis=1)

            stats[agent_name] = {
                'overall_mean': np.mean(mean_rewards),
                'overall_std': np.std(mean_rewards),
                'final_mean': np.mean(results_array[:, -1]),
                'final_std': np.std(results_array[:, -1])
            }

        return stats