import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.agent_base import BaseAgent
from agent.agent_dqn import ReplayBuffer
from models.actor_critic_network import Actor, Critic


class AgentDDPG(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, env, config):
        super().__init__(env, config)
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.device      = self.config['device']
        self.seed = random.seed(self.config["seed"])
        # self.seed        = torch.manual_seed(self.config['seed'])

        self.critic_lr          = self.config['learning_rate_critic']
        self.actor_lr           = self.config['learning_rate_actor']
        self.weight_decay           = self.config['weight_decay']

        self.GAMMA              = self.config['discount_factor']
        self.TAU                = self.config['tau']
        self.BATCH_SIZE         = self.config['batch_size']
        self.BUFFER_SIZE        = self.config['buffer_size']
        self.UPDATE_EVERY       = self.config['update_every']

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.n_states, self.n_actions).to(self.device)
        self.actor_target = Actor(self.n_states, self.n_actions).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_target = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(self.n_actions)

        # Replay memory
        self.memory = ReplayBuffer(self.config)

        # Track learning progress
        self.training_error = {
            "actor_loss": [],
            "critic_loss": []
        }

    def update(self, state, info, action, reward, done, next_state):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, done, next_state)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def select_action(self, state, info, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, dones, next_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

        self.training_error["actor_loss"].append(actor_loss.detach().cpu().item())
        self.training_error["critic_loss"].append(critic_loss.detach().cpu().item())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def save_checkpoint(self):
        model_actor_savefile = f"./models/{self.config['name']}_actor_checkpoint.pth"
        torch.save(self.actor_local.state_dict(), model_actor_savefile)
        model_critic_savefile = f"./models/{self.config['name']}_critic_checkpoint.pth"
        torch.save(self.critic_local.state_dict(), model_critic_savefile)
        print(f"Model saved to {model_critic_savefile} / {model_actor_savefile}")

    def load_checkpoint(self):
        self.actor_local.load_state_dict(torch.load(f"./models/{self.config['name']}_actor_checkpoint.pth"))
        self.critic_local.load_state_dict(torch.load(f"./models/{self.config['name']}_critic_checkpoint.pth"))



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

