import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.agent_base import BaseAgent
from models.qnetwork import QNetwork


class AgentDQN(BaseAgent):
    """Vanilla DQN (Mnih et al., 2015)."""

    def __init__(self, env, config):
        super().__init__(env, config)
        self.seed        = self.config['seed']
        self.device      = self.config['device']

        # Exploration parameters
        self.epsilon =              self.config["start_epsilon"]
        self.epsilon_decay =        self.config["epsilon_decay"]
        self.final_epsilon =        self.config["final_epsilon"]

        self.LR                 = self.config['learning_rate']
        self.GAMMA              = self.config['discount_factor']
        self.TAU                = self.config['tau']
        self.BATCH_SIZE         = self.config['batch_size']
        self.BUFFER_SIZE        = self.config['buffer_size']
        self.UPDATE_EVERY       = self.config['update_every']

        self.qnetwork_local  = QNetwork(self.n_states, self.n_actions, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(self.n_states, self.n_actions, self.seed).to(self.device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(self.config)
        self.t_step = 0


    def select_action(self, state, info):
        """Epsilon-greedy action selection."""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return int(action_values.cpu().argmax())
        return random.choice(range(self.n_actions))


    def update(self, state, info, action, reward, done, next_state):
        self.memory.add(state, action, reward, done, next_state)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            self.learn(self.memory.sample(), self.GAMMA)


    def learn(self, experiences, gamma):
        states, actions, rewards, dones, next_states = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())

    def soft_update(self, local_model, target_model):
        """theta_target = tau*theta_local + (1-tau)*theta_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


    def save_checkpoint(self):
        model_savefile = f"./models/{self.config['name']}_checkpoint.pth"
        torch.save(self.qnetwork_local.state_dict(), model_savefile)
        print(f"Model saved to {model_savefile}")

    def load_checkpoint(self):
        self.qnetwork_local.load_state_dict(torch.load(f"./models/{self.config['name']}_checkpoint.pth"))

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    """Uniform random experience replay buffer."""

    def __init__(self, config):
        self.config = config

        self.memory      = deque(maxlen=self.config["buffer_size"])
        self.batch_size  = self.config["batch_size"]

        self.experience  = namedtuple(
            "Experience", field_names=["state", "action", "reward", "done", "next_state"]
        )

        self.device = self.config['device']

    def add(self, state, action, reward, done, next_state):
        self.memory.append(self.experience(state, action, reward, done, next_state))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences])).float().to(self.device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences])).long().to(self.device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences]).astype(np.uint8)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)
