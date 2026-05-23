class AgentConfig:
    def __init__(self):
        pass



class EnvConfig:
    def __init__(self):
        self.name = ""
        self.n_episodes = 1_000
        self.seed = 0

        self.test_show = False
        self.test_n_episodes = 1_000
        # self.test_show = True
        # self.test_n_episodes = 5



if __name__ == "__main__":
    SEED = 123

    config_env = EnvConfig()
    config_env.name = "LunarLander-v3"
    config_env.n_episodes = 2_000
    config_env.seed = SEED

    config_env.test_show = True
    config_env.test_n_episodes = 1_000