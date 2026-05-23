import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal

from agent.agent_base import BaseAgent
from models.policy_network import Policy_Network


class AgentReinforce(BaseAgent):
    """REINFORCE algorithm."""

    def __init__(self, env, config):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        REINFORCE aims to maximize the Monte-Carlo returns.
        REINFORCE is an acronym for “ ‘RE’ward ‘I’ncrement ‘N’on-negative ‘F’actor times ‘O’ffset ‘R’einforcement times ‘C’haracteristic ‘E’ligibility


        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__(env, config)
        self.policy_type = self.config["policy_type"]

        self.probs = []         # Stores probability values of the sampled action
        self.rewards = []       # Stores the corresponding rewards

        self.net = Policy_Network(self.n_states, self.n_actions, policy_type=self.policy_type)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.LR)


    def select_action(self, state, info) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))

        if self.policy_type == "discrete":
            logits = self.net(state)
            dist = Categorical(logits=logits + self.epsilon)
            action = dist.sample()
            self.probs.append(dist.log_prob(action))
            return action.item()   # CartPole needs an int

        else:
            mean, std = self.net(state)
            dist = Normal(mean + self.epsilon, std+ self.epsilon)
            action = dist.sample()

            # store summed log-prob for multi-dim continuous actions
            self.probs.append(dist.log_prob(action).sum(dim=-1))

            # convert to numpy with correct shape for the env
            action = action.squeeze(0).detach().cpu().numpy()

            # optional: clip to env bounds
            low = self.env.action_space.low
            high = self.env.action_space.high
            action = np.clip(action, low, high)

            return action

    def update(self, state, info, action, reward, terminated, next_state):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        self.rewards.append(reward)

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.discount_factor * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(self.probs).squeeze()

        # Update the loss with the mean log probability and deltas
        # Now, we compute the correct total loss by taking the sum of the element-wise products.
        loss = -torch.sum(log_probs * deltas)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

        self.training_error.append(loss.detach().cpu().item())


    def save_checkpoint(self):
        model_savefile = f"./models/{self.config['name']}_checkpoint.pth"
        torch.save(self.net.state_dict(), model_savefile)
        print(f"Model saved to {model_savefile}")

    def load_checkpoint(self):
        self.net.load_state_dict(torch.load(f"./models/{self.config['name']}_checkpoint.pth"))