# We will implement an Advantage Actor-Critic from scratch to look at how you can feed batched states into your networks
# to get a vector of actions (one action per environment) and calculate the losses for actor and critic on minibatches of transitions.
# Each minibatch contains the transitions of one sampling phase: n_steps_per_update steps are executed in n_envs environments in parallel
# (multiply the two to get the number of transitions in a minibatch). After each sampling phase, the losses are calculated
# and one gradient step is executed. To calculate the advantages, we are going to use the Generalized Advantage Estimation (GAE) method [2],
# which balances the tradeoff between variance and bias of the advantage estimates.
#
# The A2C agent class is initialized with the number of features of the input state, the number of actions the agent can take,
# the learning rates and the number of environments that run in parallel to collect experiences. The actor and critic networks
# are defined and their respective optimizers are initialized. The forward pass of the networks takes in a batched vector of states
# and returns a tensor of state values and a tensor of action logits. The select_action method returns a tuple of the chosen actions,
# the log-probs of those actions, and the state values for each action. In addition, it also returns the entropy of the policy distribution,
# which is subtracted from the loss later (with a weighting factor ent_coef) to encourage exploration.
#
# The get_losses function calculates the losses for the actor and critic networks (using GAE), which are then updated using the update_parameters function.

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym

from a2c import A2C
from params import randomize_domain, n_envs, critic_lr, actor_lr, n_updates, n_steps_per_update, gamma, lam, ent_coef, save_weights, actor_weights_path, critic_weights_path, load_weights


def create_env():
    # When you calculate the losses for the two Neural Networks over only one epoch,
    # it might have a high variance. With vectorized environments,
    # we can play with n_envs in parallel and thus get up to a linear speedup
    # (meaning that in theory, we collect samples n_envs times quicker)
    # that we can use to calculate the loss for the current policy and critic network.
    #
    # The simplest way to create vector environments is by calling gym.vector.make,
    # which creates multiple instances of the same environment:
    # envs = gym.make_vec("LunarLander-v3", num_envs=3, max_episode_steps=600)


    # Manually setting up 3 parallel ‘LunarLander-v3’ envs with different parameters:
    # envs = gym.vector.SyncVectorEnv(
    #     [
    #         lambda: gym.make(
    #             "LunarLander-v3",
    #             gravity=-10.0,
    #             enable_wind=True,
    #             wind_power=15.0,
    #             turbulence_power=1.5,
    #             max_episode_steps=600,
    #         ),
    #         lambda: gym.make(
    #             "LunarLander-v3",
    #             gravity=-9.8,
    #             enable_wind=True,
    #             wind_power=10.0,
    #             turbulence_power=1.3,
    #             max_episode_steps=600,
    #         ),
    #         lambda: gym.make(
    #             "LunarLander-v3", gravity=-7.0, enable_wind=False, max_episode_steps=600
    #         ),
    #     ]
    # )

    # Randomly generating the parameters for 3 parallel ‘LunarLander-v3’ envs,
    # using np.clip to stay in the recommended parameter space:

    # envs = gym.vector.SyncVectorEnv(
    #     [
    #         lambda: gym.make(
    #             "LunarLander-v3",
    #             gravity=np.clip(
    #                 np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
    #             ),
    #             enable_wind=np.random.choice([True, False]),
    #             wind_power=np.clip(
    #                 np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
    #             ),
    #             turbulence_power=np.clip(
    #                 np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
    #             ),
    #             max_episode_steps=600,
    #         )
    #         for i in range(3)
    #     ]
    # )


    # environment setup
    if randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLander-v3",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                )
                for i in range(n_envs)
            ]
        )

    else:
        envs = gym.make_vec("LunarLander-v3", num_envs=n_envs, max_episode_steps=600)

    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.n

    return envs, obs_shape, action_shape



def train(envs_wrapper, agent, device):
    critic_losses = []
    actor_losses = []
    entropies = []

    # use tqdm to get a progress bar for training
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode (sample phase)
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_entropies = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)

        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                actions.cpu().numpy()
            )

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs
            ep_entropies[step] = entropy

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            ep_entropies,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(ep_entropies.detach().mean().cpu().numpy())

    return actor_losses, critic_losses, entropies

def plot(envs_wrapper, agent, actor_losses, critic_losses, entropies):
    """ plot the results """

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the LunarLander-v3 environment \n \
                 (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
            np.convolve(
                np.array(envs_wrapper.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
            np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
            np.convolve(
                np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
            np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()



def test_agent(agent):
    """ play a couple of showcase episodes """

    n_showcase_episodes = 3

    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")

        # create a new sample environment to get new random parameters
        if randomize_domain:
            env = gym.make(
                "LunarLander-v3",
                render_mode="human",
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=500,
            )
        else:
            env = gym.make("LunarLander-v3", render_mode="human", max_episode_steps=500)

        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                action, _, _, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(action.item())

            # update if the environment is done
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    # set the device
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    envs, obs_shape, action_shape = create_env()

    # init the agent
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    # create a wrapper environment to save episode returns and episode lengths
    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
        envs, buffer_length=n_envs * n_updates
    )


    actor_losses, critic_losses, entropies = train(envs_wrapper, agent, device)

    """ save network weights """
    if save_weights:
        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)


    plot(envs_wrapper, agent, actor_losses, critic_losses, entropies)

    """ load network weights """
    if load_weights:
        agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr)

        agent.actor.load_state_dict(torch.load(actor_weights_path))
        agent.critic.load_state_dict(torch.load(critic_weights_path))
        agent.actor.eval()
        agent.critic.eval()

    test_agent(agent)