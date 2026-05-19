from experiment_creator import ExperimentCreator

base_config_env = {
    "name"              : "FrozenLake-v1",
    "is_slippery"       : False,
    "proba_frozen"      : 0.9,

    "n_episodes"        : 2_000,                                # Number of hands to practice

    "seed"              : 123,
}

base_config_agent = {
    "name"              : "q_learning",
    "learning_rate"     : 0.8,                                  # How fast to learn (higher = faster but less stable)
    "discount_factor"   : 0.95,                                 # Always keep some exploration
    "start_epsilon"     : 1.0,                                  # Start with 100% random actions
    "final_epsilon"     : 0.1,                                  # Always keep some exploration
    "use_action_mask"   : False,
}

base_config_agent.update({
    "epsilon_decay"   : base_config_agent["start_epsilon"] / (base_config_env["n_episodes"] / 2)  ,    # Reduce exploration over time
})

# Training hyperparameters
configs_env = [{
    **base_config_env,
    "map_size": map_size,
} for map_size in [4,5,7,9,11] ]

# experiment_creator = ExperimentCreator(
#     savefolder=f"./plots/{base_config_env["name"]}/train/",
#     savefile=f"baseline",
#     configs_env=configs_env, configs_agent=base_config_agent
# )
# experiment_creator.start_experiment()
# experiment_creator.plot()