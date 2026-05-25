import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent.agent_base import BaseAgent
from models.actor_critic_network import ActorCriticNetwork



class AgentA2C(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.seed = torch.manual_seed(self.config['seed'])
        self.device = self.config['device']

        self.ACTOR_LR = self.config.get('learning_rate_actor', self.config.get('learning_rate', 3e-4))
        self.CRITIC_LR = self.config.get('learning_rate_critic', self.config.get('learning_rate', 3e-4))
        self.GAMMA = self.config['discount_factor']
        self.LAM = self.config.get('lam', 0.95)
        self.VF_COEF = self.config.get('vf_coef', 0.5)
        self.ENTROPY_COEF = self.config.get('ent_coef', 0.01)
        self.N_STEPS = self.config.get('n_steps', 5)
        self.MAX_GRAD_NORM = self.config.get('max_grad_norm', 0.5)

        network_layers = self.config.get('layers', [64, 64])
        self.ac_network = ActorCriticNetwork(
            self.n_states,
            self.n_actions,
            layers=network_layers
        ).to(self.device)

        # Safer than separate optimizers, especially if the network has a shared trunk
        # lr = self.config.get('learning_rate', 3e-4)
        # self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)

        self.actor_params = []
        self.critic_params = []

        self.optimizer = optim.Adam([
            {'params': self.ac_network.shared_layers.parameters(), 'lr': self.config.get('learning_rate', 3e-4)},
            {'params': self.ac_network.actor.parameters(), 'lr': self.ACTOR_LR},
            {'params': self.ac_network.critic.parameters(), 'lr': self.CRITIC_LR},
        ])

        self.rollout_buffer = RolloutBuffer(self.config)

        self.training_error = {
            'total_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': []
        }

    def select_action(self, state, info):
        state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        self.ac_network.eval()
        with torch.no_grad():
            logits, _ = self.ac_network(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()  # ← ALWAYS sample (stochastic policy)

        self.ac_network.train()
        return int(action.item())

    def update(self, state, info, action, reward, done, next_state):
        self.rollout_buffer.add(state, action, reward, done, next_state)

        if len(self.rollout_buffer) >= self.N_STEPS or done:
            self.learn()
            self.rollout_buffer.clear()

    def learn(self):
        if len(self.rollout_buffer) == 0:
            return

        states, actions, rewards, dones, next_states = self.rollout_buffer.get()
        T = len(rewards)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        masks = 1.0 - dones_tensor

        with torch.no_grad():
            # Get value estimates
            _, old_values = self.ac_network(states_tensor)
            old_values = old_values.squeeze(-1)

            # Bootstrap from final state
            last_next_state = torch.as_tensor(
                next_states[-1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, bootstrap_value = self.ac_network(last_next_state)
            bootstrap_value = bootstrap_value.squeeze(-1).squeeze(0)

            # If episode ended, bootstrap value should be 0
            if dones[-1]:
                bootstrap_value = torch.tensor(0.0, device=self.device)

            # Compute returns via bootstrapping (for critic target)
            returns = torch.zeros(T, dtype=torch.float32, device=self.device)
            R = bootstrap_value
            for t in reversed(range(T)):
                R = rewards_tensor[t] + self.GAMMA * R * masks[t]
                returns[t] = R

            # Compute advantages using GAE
            values_ext = torch.cat([old_values, bootstrap_value.view(1)])
            advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
            gae = 0.0

            for t in reversed(range(T)):
                delta = rewards_tensor[t] + self.GAMMA * values_ext[t + 1] * masks[t] - values_ext[t]
                gae = delta + self.GAMMA * self.LAM * masks[t] * gae
                advantages[t] = gae

            # Store unnormalized for logging
            advantages_unnorm = advantages.clone()

            # Normalize advantages (for actor loss only)
            if advantages.numel() > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-4:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if returns.numel() > 1:
                returns_mean = returns.mean()
                returns_std = returns.std()
                if returns_std > 1e-4:
                    returns = (returns - returns_mean) / (returns_std + 1e-8)

        # Compute NEW predictions
        logits, new_values = self.ac_network(states_tensor)
        new_values = new_values.squeeze(-1)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy()

        # Compute losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(new_values, returns)  # Use returns, not advantages!
        entropy_bonus = entropy.mean()

        total_loss = actor_loss + self.VF_COEF * critic_loss - self.ENTROPY_COEF * entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()

        self.training_error['total_loss'].append(total_loss.item())
        self.training_error['actor_loss'].append(actor_loss.item())
        self.training_error['critic_loss'].append(critic_loss.item())
        self.training_error['entropy'].append(entropy_bonus.item())

        # Better logging
        if len(self.training_error['total_loss']) % 100 == 0:
            print(f"Loss: {total_loss.item():.4f} | Actor: {actor_loss.item():.4f} | "
                  f"Critic: {critic_loss.item():.4f} | Entropy: {entropy_bonus.item():.4f} | "
                  f"Avg Adv (unnorm): {advantages_unnorm.mean().item():.4f} | "
                  f"Avg Return: {returns.mean().item():.4f} | Avg Reward: {rewards_tensor.mean().item():.4f}")

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
    Stores on-policy trajectories for A2C learning with GAE.

    Unlike DQN's replay buffer which stores and samples randomly,
    this buffer stores recent consecutive transitions and is cleared
    after each learning update (on-policy learning).

    Additionally stores log_probs, state_values, and entropies computed
    during action selection to avoid recomputation.
    """

    def __init__(self, config):
        self.config = config
        self.n_steps = self.config.get('n_steps', 5)
        self.device = config['device']

        # Transition data
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
        self.dones.append(float(done))  # Convert bool to float
        self.next_states.append(next_state)


    def get(self):
        """
        Return all stored transitions and action info as numpy arrays.
        Called when buffer is full or episode ends.
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.next_states),
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




