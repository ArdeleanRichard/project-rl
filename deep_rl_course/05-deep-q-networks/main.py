"""
Excellent results with this q-learning,
- but these environments were relatively simple because the state space was discrete and small (16 different states for FrozenLake-v1 and 500 for Taxi-v3). 
- For comparison, the state space in Atari games can contain 10^9 to 10^11 states.

But as we’ll see, producing and updating a Q-table can become ineffective in large state space environments.

So in this unit, we’ll study our first Deep Reinforcement Learning agent: Deep Q-Learning. 
- Instead of using a Q-table, Deep Q-Learning uses a Neural Network that takes a state and approximates Q-values for each action based on that state.
"""

from dqn_agent import AgentDQN, AgentDoubleDQN, AgentPriorityDQN, AgentDuelingDQN, AgentDistributionalDQN, AgentNoisyDQN, AgentRainbow
from dqn import prep_env, dqn, plot_scores, test_agent

def run_dqn():
    env = prep_env()
    agent = AgentDQN(state_size=8, action_size=4, seed=0)
    # watch_agent(env, agent)

    savefile = "./models/dqn_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(agent, savefile=savefile)



def run_dqn_double():
    env = prep_env()
    agent = AgentDoubleDQN(state_size=8, action_size=4, seed=0)

    savefile = "./models/dqn_double_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(agent, savefile=savefile)


def run_dqn_priority():
    env = prep_env()
    agent = AgentPriorityDQN(state_size=8, action_size=4, seed=0)

    savefile = "./models/dqn_priority_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(agent, savefile=savefile)


def run_dqn_dueling():
    env = prep_env()
    agent = AgentDuelingDQN(state_size=8, action_size=4, seed=0)

    savefile = "./models/dqn_dueling_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)

    test_agent(agent, savefile=savefile)


def run_dqn_distributional():
    """
    C51: predicts a probability distribution over returns instead of a scalar.
    Selects actions by argmax of expected Q = Σ z_i · p_i.
    Uses ε-greedy (like standard DQN) since there is no built-in noise.
    """
    env    = prep_env()
    agent  = AgentDistributionalDQN(state_size=8, action_size=4, seed=0)

    savefile = "./models/dqn_distributional_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile)
    plot_scores(scores)
    test_agent(agent, savefile)


def run_dqn_noisy():
    """
    Noisy DQN: NoisyLinear layers provide intrinsic, state-dependent exploration.
    No ε schedule is needed — pass eps_start=0 so the loop never applies ε-greedy.
    """
    env   = prep_env()
    agent = AgentNoisyDQN(state_size=8, action_size=4, seed=0)

    savefile = "./models/dqn_noisy_checkpoint.pth"
    # eps_start=0: the noisy network handles all exploration internally
    scores = dqn(env, agent, savefile=savefile, eps_start=0.0, eps_end=0.0, eps_decay=1.0)
    plot_scores(scores)
    test_agent(agent, savefile)


def run_rainbow():
    """
    Rainbow: Double + Prioritized + Dueling + Multi-step + Distributional + Noisy.
    No ε schedule — NoisyLinear handles exploration.
    """
    env   = prep_env()
    agent = AgentRainbow(state_size=8, action_size=4, seed=0, n_steps=3)


    savefile = "./models/dqn_rainbow_checkpoint.pth"
    scores = dqn(env, agent, savefile=savefile, eps_start=0.0, eps_end=0.0, eps_decay=1.0)
    plot_scores(scores)
    test_agent(agent, savefile)



if __name__ == "__main__":
    # run_dqn()
    # run_dqn_double()
    # run_dqn_priority()
    # run_dqn_dueling()
    # run_dqn_distributional()
    # run_dqn_noisy()
    run_rainbow()

    pass

