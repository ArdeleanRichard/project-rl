import torch

from creator_agent import AgentCreator
from creator_environment import EnvironmentCreator
from creator_trainer import Trainer

SEED = 123

config_env = {
    "name"              : "CartPole-v1",
    "n_episodes"        : 1_000,                                # Number of episodes to practice

    "seed": SEED,

    # test_env configuration
    "test_show"          : False,
    "test_n_episodes"    : 1_000,
    # "test_show"          : True,
    # "test_n_episodes"    : 5,

    # threshold
    "score_threshold"   : 195.0,
}


config_agent = {
    "name"              : "reinforce",

    "learning_rate"     : 1e-4,                                 # How fast to learn (higher = faster but less stable)
    "discount_factor"   : 1.0,                                  # Always keep some exploration
    "start_epsilon"     : 1e-4,                                 # Start with 100% random actions

    "device"            : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": SEED,
}


# Create environment
env_creator = EnvironmentCreator(config_env)
env = env_creator.create()
env_creator.print_info()

# Create agent
agent = AgentCreator(env=env, config=config_agent).create()

# Create trainer
trainer = Trainer(env=env, agent=agent)
trainer.train_agent()
trainer.save_agent()
trainer.plot_train(f"./plots/{config_env["name"]}/train/")
trainer.test_agent()

env.close()