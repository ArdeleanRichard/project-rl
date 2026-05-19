"""
https://gymnasium.farama.org/introduction/train_agent/
Q-learning builds a giant “cheat sheet” called a Q-table that tells the agent how good each action is in each situation:
    Rows = different situations (states) the agent can encounter
    Columns = different actions the agent can take
    Values = how good that action is in that situation (expected future reward)

For Blackjack:
    States: Your hand value, dealer’s showing card, whether you have a usable ace
    Actions: Hit (take another card) or Stand (keep current hand)
    Q-values: Expected reward for each action in each state

The Learning Process
Try an action and see what happens (reward + new state)
    Update your cheat sheet: “That action was better/worse than I thought”
    Gradually improve by trying actions and updating estimates
    Balance exploration vs exploitation: Try new things vs use what you know works
"""
import gymnasium as gym
from agent import Agent
from q_learning import train_agent, test_agent
from plot import plot_train, plot_policy

# Training hyperparameters
n_episodes      = 100_000                               # Number of hands to practice
learning_rate   = 0.001                                 # How fast to learn (higher = faster but less stable)
start_epsilon   = 1.0                                   # Start with 100% random actions
epsilon_decay   = start_epsilon / (n_episodes / 2)      # Reduce exploration over time
final_epsilon   = 0.1                                   # Always keep some exploration

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


train_agent(env, agent, n_episodes)

"""
Interpreting the Results
    Reward Plot: Should show gradual improvement from ~-0.05 (slightly negative) to ~-0.01 (near optimal). Blackjack is a difficult game - even perfect play loses slightly due to the house edge.
    Episode Length: Should stabilize around 2-3 actions per episode. Very short episodes suggest the agent is standing too early; very long episodes suggest hitting too often.
    Training Error: Should decrease over time, indicating the agent’s predictions are getting more accurate. Large spikes early in training are normal as the agent encounters new situations.
"""
plot_train(env, agent)
plot_policy(agent)

"""
Good Blackjack performance:
    Win rate: 42-45% (house edge makes >50% impossible)
    Average reward: -0.02 to +0.01
    Consistency: Low standard deviation indicates reliable strategy
"""
test_agent(agent, env)


"""
Common Training Issues and Solutions
🚨 Agent Never Improves
    Symptoms: Reward stays constant, large training errors Causes: Learning rate too high/low, poor reward design, bugs in update logic Solutions:
    Try learning rates between 0.001 and 0.1
    Check that rewards are meaningful (-1, 0, +1 for Blackjack)
    Verify Q-table is actually being updated

🚨 Unstable Training
    Symptoms: Rewards fluctuate wildly, never converge Causes: Learning rate too high, insufficient exploration Solutions:
    Reduce learning rate (try 0.01 instead of 0.1)
    Ensure minimum exploration (final_epsilon ≥ 0.05)
    Train for more episodes

🚨 Agent Gets Stuck in Poor Strategy
    Symptoms: Improvement stops early, suboptimal final performance Causes: Too little exploration, learning rate too low Solutions:
    Increase exploration time (slower epsilon decay)
    Try higher learning rate initially
    Use different exploration strategies (optimistic initialization)

🚨 Learning Too Slow
    Symptoms: Agent improves but very gradually Causes: Learning rate too low, too much exploration Solutions:
    Increase learning rate (but watch for instability)
    Faster epsilon decay (less random exploration)
    More focused training on difficult states
"""

env.close()