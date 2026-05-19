from experiment_creator import ExperimentCreator

BASE_RANDOM_SEED = 58922320
n_runs = 10

base_config_env = {
    "name"              : "Taxi-v3",
    "n_episodes"        : 5_000,                                # Number of hands to practice
}

base_config_agent = {
    "name": "q_learning",
    "learning_rate": 0.1,  # How fast to learn (higher = faster but less stable)
    "discount_factor": 0.95,  # Always keep some exploration
    "start_epsilon": 1.0,  # Start with 100% random actions
    "final_epsilon": 0.1,  # Always keep some exploration
    "use_action_mask"   : False,
}

base_config_agent.update({
    "epsilon_decay"   : base_config_agent["start_epsilon"] / (base_config_env["n_episodes"] / 2)  ,    # Reduce exploration over time
})

# Training hyperparameters
configs_env = [{
    **base_config_env,
    "seed": BASE_RANDOM_SEED + i,
} for i in range(n_runs) ]


experiment_creator = ExperimentCreator(
    savefolder=f"./plots/{base_config_env["name"]}/train/",
    savefile=f"baseline",
    configs_env=configs_env, configs_agent=base_config_agent
)
experiment_creator.start_experiment()
experiment_creator.plot()