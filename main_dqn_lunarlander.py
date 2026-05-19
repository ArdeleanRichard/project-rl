import torch

from agent_creator import AgentCreator
from environment_creator import EnvironmentCreator
from trainer import Trainer

SEED = 123

config_env = {
    "name"              : "LunarLander-v3",
    "n_episodes"        : 2_000,                                # Number of episodes to practice

    "seed": SEED,

    # test_env configuration
    "test_show"          : False,
    "test_n_episodes"    : 1_000,
    # "test_show"          : True,
    # "test_n_episodes"    : 5,

    # threshold
    "score_threshold"   : 200,
}


config_agent = {
    "name"              : "dqn",

    "max_t"             : 1000,
    "learning_rate"     : 5e-4,                                 # How fast to learn (higher = faster but less stable)
    "discount_factor"   : 0.99,                                 # Always keep some exploration
    "start_epsilon"     : 1.0,                                  # Start with 100% random actions
    "final_epsilon"     : 0.01,
    "epsilon_decay"     : 0.995  ,                              # Reduce exploration over time

    "batch_size"        : 64,                                   # minibatch size
    "buffer_size"       : int(1e5),                             # replay buffer size
    "tau"               : 1e-3,                                 # for soft update of target parameters
    "update_every"      : 4,                                    # how often to update the network

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