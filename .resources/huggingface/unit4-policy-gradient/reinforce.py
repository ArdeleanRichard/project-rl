import numpy as np
from collections import deque

# PyTorch
import torch

# Gym
import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_env(env_id):
    # Create the env
    env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample())  # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    return env, s_size, a_size



def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    """
    Pseudocode:
        1: procedure REINFORCE
        2:  Start with policy model π0
        3:  repeat:
        4:      Generate an episode S0, A0, r0, ... ST, AT, rT
        5:      for t from T to 0:
        6:          Gt = sum(gamma * rk) where k=t -> T (expected returns)
        7:      L(θ) = 1/T sum (Gt log πθ(At|St)) where t = 0->T
        8:      optimize πθ using derivative of L(θ)
        9: end procedure
    """
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []

    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ =  env.reset() # TODO: reset the environment

        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state) # TODO get the action
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action) # TODO: take an env step

            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft( gamma * disc_return_t + rewards[t])  # TODO: complete here

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')

    return scores


def evaluate_agent(env_id, policy, n_eval_episodes, max_steps):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param policy: The Reinforce agent
    :param n_eval_episodes: Number of episode to evaluate the agent
    """

    env = gym.make(env_id, render_mode='human')
    state, _ = env.reset()

    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

