from agent import Agent
from monitor import interact
import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v4')

print("-"*60)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print("-"*60)
print()

agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)