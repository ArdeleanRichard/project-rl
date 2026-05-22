import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent.agent_base import BaseAgent
from models.actor_critic_network import ActorCriticNetwork


class AgentA2C(BaseAgent):
    """
    Advantage Actor-Critic (A2C) Agent.

    A2C is a synchronous version of A3C that uses:
    - Actor: learns a policy π(a|s)
    - Critic: learns a value function V(s)
    - Advantage: A(s,a) = Q(s,a) - V(s) ≈ R + γV(s') - V(s)

    The actor is trained to maximize expected advantage, while the critic
    is trained to minimize the TD error. Uses n-step returns for better
    credit assignment than 1-step TD.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.seed = torch.manual_seed(self.config['seed'])
        self.device = self.config['device']

        # Learning parameters
        self.LR = self.config['learning_rate']
        self.GAMMA = self.config['discount_factor']
        self.VALUE_COEF = self.config.get('value_loss_coef', 0.5)       # Critic loss weight
        self.ENTROPY_COEF = self.config.get('entropy_coef', 0.01)       # Exploration bonus
        self.N_STEPS = self.config.get('n_steps', 5)                    # n-step returns
        self.MAX_GRAD_NORM = self.config.get('max_grad_norm', 0.5)      # Gradient clipping

        # Network
        network_layers = self.config.get('layers', [64, 64])
        self.ac_network = ActorCriticNetwork(
            self.n_states,
            self.n_actions,
            layers=network_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.LR)

        # Storage for n-step rollouts
        self.rollout_buffer = RolloutBuffer(self.config, self.N_STEPS)

        # Tracking for separate loss components
        self.training_error = {
            'total_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': []
        }

    def select_action(self, state, info):
        """
        Sample action from the policy distribution π(a|s).
        During training, samples stochastically for exploration.
        During testing (epsilon=0), uses greedy argmax.
        """
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

        self.ac_network.eval()
        with torch.no_grad():
            action_logits, _ = self.ac_network(state)
            action_probs = F.softmax(action_logits, dim=-1)
        self.ac_network.train()

        # Greedy action for testing
        if self.epsilon == 0.0:
            return int(action_probs.argmax())

        # Stochastic sampling for training (exploration)
        dist = Categorical(action_probs)
        action = dist.sample()
        return int(action.item())

    def update(self, state, info, action, reward, done, next_state):
        """
        Store transition in rollout buffer.
        When buffer is full or episode ends, perform learning update.
        """
        self.rollout_buffer.add(state, action, reward, done, next_state)

        # Learn when we have n steps OR episode ends
        if len(self.rollout_buffer) >= self.N_STEPS or done:
            self.learn()
            self.rollout_buffer.clear()

    def learn(self):
        """
        Update actor and critic using n-step returns.

        Actor loss: -log π(a|s) * A(s,a) - β * H(π)
        Critic loss: MSE between V(s) and n-step return

        Where:
        - A(s,a) is the advantage (how much better than average)
        - H(π) is entropy (encourages exploration)
        - β controls exploration strength
        """
        if len(self.rollout_buffer) == 0:
            return

        # Get stored transitions
        states, actions, rewards, dones, next_states = self.rollout_buffer.get()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Compute n-step returns: R_t = r_t + γr_{t+1} + ... + γ^n V(s_{t+n})
        with torch.no_grad():
            _, next_values = self.ac_network(next_states)
            returns = rewards + self.GAMMA * next_values * (1 - dones)

            # For n-step: accumulate discounted rewards backwards
            n_step_returns = torch.zeros_like(rewards)
            running_return = returns[-1]
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.GAMMA * running_return * (1 - dones[t])
                n_step_returns[t] = running_return

        # Forward pass
        action_logits, state_values = self.ac_network(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        # Compute advantages: A(s,a) = R - V(s)
        advantages = n_step_returns - state_values

        # Actor loss: policy gradient with advantage
        log_probs = dist.log_prob(actions.squeeze())
        actor_loss = -(log_probs.unsqueeze(1) * advantages.detach()).mean()

        # Critic loss: TD error
        critic_loss = F.mse_loss(state_values, n_step_returns)

        # Entropy bonus: encourages exploration
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = actor_loss + self.VALUE_COEF * critic_loss - self.ENTROPY_COEF * entropy

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()

        # Track losses
        self.training_error['total_loss'].append(total_loss.detach().cpu().item())
        self.training_error['actor_loss'].append(actor_loss.detach().cpu().item())
        self.training_error['critic_loss'].append(critic_loss.detach().cpu().item())
        self.training_error['entropy'].append(entropy.detach().cpu().item())

    def save_checkpoint(self):
        model_savefile = f"./models/{self.config['name']}_checkpoint.pth"
        torch.save(self.ac_network.state_dict(), model_savefile)
        print(f"Model saved to {model_savefile}")

    def load_checkpoint(self):
        self.ac_network.load_state_dict(torch.load(f"./models/{self.config['name']}_checkpoint.pth"))

    def decay_epsilon(self):
        """
        A2C doesn't use epsilon-greedy exploration.
        Exploration comes from stochastic policy + entropy bonus.
        This method is kept for compatibility with the training loop.
        """
        pass


class RolloutBuffer:
    """
    Stores n-step trajectories for A2C learning.

    Unlike DQN's replay buffer which stores and samples randomly,
    this buffer stores recent consecutive transitions and is cleared
    after each learning update (on-policy learning).
    """

    def __init__(self, config, n_steps):
        self.config = config
        self.n_steps = n_steps
        self.device = config['device']

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def add(self, state, action, reward, done, next_state):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def get(self):
        """
        Return all stored transitions as numpy arrays.
        Called when buffer is full or episode ends.
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.next_states)
        )

    def clear(self):
        """Clear the buffer after learning update."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.states)









