import gymnasium as gym
from gymnasium import Wrapper

from utils.utils import get_space_size


class EnvWrapper(Wrapper):
    def __init__(self, env, config=None):
        super().__init__(env)
        self.config = config or {}

    def reset(self, **kwargs):
        # You can use self.config here
        return self.env.reset(**kwargs)

    def step(self, action):
        # Access config in step if needed
        return self.env.step(action)

    # Forward missing attributes to wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)

class EnvironmentCreator:
    def __init__(self, config=None):
        self.config = config
        self.env = None

    def create(self):
        if self.config["name"] == "Blackjack-v1":
            env = gym.make("Blackjack-v1", render_mode="rgb_array", sab=False)
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=self.config["n_episodes"])

            self.env = EnvWrapper(env, self.config)
            return self.env

        if self.config["name"] == "FrozenLake-v1":
            from gymnasium.envs.toy_text.frozen_lake import generate_random_map

            env = gym.make(
                "FrozenLake-v1",
                render_mode="rgb_array",
                is_slippery=self.config["is_slippery"],
                desc=generate_random_map(
                    size=self.config["map_size"], p=self.config["proba_frozen"], seed=self.config["seed"]
                ),
            )
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=self.config["n_episodes"])

            self.env = EnvWrapper(env, self.config)
            return self.env

        if self.config["name"] == "LunarLander-v3":
            env = gym.make('LunarLander-v3', render_mode="rgb_array")
            obs, info = env.reset(seed=self.config["seed"])
            env.action_space.seed(self.config["seed"])
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=self.config["n_episodes"])

            self.env = EnvWrapper(env, self.config)
            return self.env

        if self.config["name"] == "Taxi-v3":
            env = gym.make('Taxi-v3', render_mode="rgb_array")
            obs, info = env.reset(seed=self.config["seed"])
            env.action_space.seed(self.config["seed"])
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=self.config["n_episodes"])

            self.env = EnvWrapper(env, self.config)
            return self.env

    @staticmethod
    def create_test_env(config):
        if config["test_show"] == True:
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        if config["name"] == "Blackjack-v1":
            env_test = gym.make("Blackjack-v1", render_mode=render_mode)
            return EnvWrapper(env_test, config)

        if config["name"] == "FrozenLake-v1":
            env_test = gym.make("FrozenLake-v1", render_mode=render_mode)
            return EnvWrapper(env_test, config)

        if config["name"] == "LunarLander-v3":
            env_test = gym.make("LunarLander-v3", render_mode=render_mode)
            return EnvWrapper(env_test, config)

        if config["name"] == "Taxi-v3":
            env_test = gym.make("Taxi-v3", render_mode=render_mode)
            return EnvWrapper(env_test, config)



    def print_info(self):
        print("-" * 60)
        print(f"Environment: {self.config['name']}")
        print("-" * 60)
        print(f"\tState space shape: {self.env.observation_space}")
        print(f"\tAction space shape: {self.env.action_space}")
        print(f"\tState space size: {get_space_size(self.env.observation_space)}")
        print(f"\tAction space size: {get_space_size(self.env.action_space)}")
        print("-" * 60)
        print()
