"""
What to Expect During Training
    Early episodes (0-10,000):
    Agent acts mostly randomly (high epsilon)
    Wins about 43% of hands (slightly worse than random due to poor strategy)
    Large learning errors as Q-values are very inaccurate

Middle episodes (10,000-50,000):
    Agent starts finding good strategies
    Win rate improves to 45-48%
    Learning errors decrease as estimates get better

Later episodes (50,000+):
    Agent converges to near-optimal strategy
    Win rate plateaus around 49% (theoretical maximum for this game)
    Small learning errors as Q-values stabilize
"""
import numpy as np
from tqdm import tqdm  # Progress bar


def train_agent(env, agent, n_episodes):
    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

# Test the trained agent
def test_agent(agent, env, n_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {n_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
