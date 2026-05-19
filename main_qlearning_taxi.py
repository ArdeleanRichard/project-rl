from agent_creator import AgentCreator
from environment_creator import EnvironmentCreator
from trainer import Trainer

# Training hyperparameters
config_env = {
    "name"              : "Taxi-v3",
    "n_episodes"        : 5_000,                                # Number of hands to practice

    # test_env configuration
    "test_show"         : False,
    "test_n_episodes"   : 1000,

    "seed": 123,
}

config_agent = {
    "name"              : "q_learning",
    "learning_rate"     : 0.1,                                  # How fast to learn (higher = faster but less stable)
    "discount_factor"   : 0.95,                                 # Always keep some exploration
    "start_epsilon"     : 1.0,                                  # Start with 100% random actions
    "final_epsilon"     : 0.1,                                  # Always keep some exploration
    "use_action_mask"   : True,
}

config_agent.update({
    "epsilon_decay"   : config_agent["start_epsilon"] / (config_env["n_episodes"] / 2)  ,    # Reduce exploration over time
})


# Create environment
env_creator = EnvironmentCreator(config_env)
env = env_creator.create()
env_creator.print_info()

# Create agent
agent = AgentCreator(env=env, config=config_agent).create()

# Create trainer
trainer = Trainer(env=env, agent=agent, config_env=config_env)
trainer.train_agent()
trainer.plot_train(f"./plots/{config_env["name"]}/train/")
trainer.plot_policy(f"./plots/{config_env["name"]}/train/")
trainer.test_agent(test_n_episodes=1000)

env.close()