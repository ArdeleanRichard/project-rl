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
        self.ENTROPY_COEF = self.config.get('ent_coef', self.config.get('entropy_coef', 0.01))
        self.N_STEPS = self.config.get('n_steps', 5)
        self.MAX_GRAD_NORM = self.config.get('max_grad_norm', 0.5)

        network_layers = self.config.get('layers', [64, 64])
        self.ac_network = ActorCriticNetwork(
            self.n_states,
            self.n_actions,
            layers=network_layers
        ).to(self.device)

        # Safer than separate optimizers, especially if the network has a shared trunk
        lr = self.config.get('learning_rate', 3e-4)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)

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

            # Keep greedy only for explicit evaluation when epsilon is set to 0
            if getattr(self, "epsilon", 1.0) == 0.0:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

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

        states, actions, rewards, dones, next_states, _, _, _ = self.rollout_buffer.get()
        T = len(rewards)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        masks = 1.0 - dones_tensor

        # Current policy/value predictions for the whole rollout
        logits, values = self.ac_network(states_tensor)
        values = values.squeeze(-1)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy()

        # Bootstrap value from the last next_state
        with torch.no_grad():
            last_next_state = torch.as_tensor(
                next_states[-1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, final_value = self.ac_network(last_next_state)
            final_value = final_value.squeeze(-1).squeeze(0)

        with torch.no_grad():
            values_ext = torch.cat([values.detach(), final_value.view(1)])

            advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
            gae = 0.0

            for t in reversed(range(T)):
                delta = rewards_tensor[t] + self.GAMMA * values_ext[t + 1] * masks[t] - values_ext[t]
                gae = delta + self.GAMMA * self.LAM * masks[t] * gae
                advantages[t] = gae

            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            returns = advantages + values_ext[:T]

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropy.mean()

        total_loss = actor_loss + critic_loss - self.ENTROPY_COEF * entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()

        self.training_error['total_loss'].append(total_loss.item())
        self.training_error['actor_loss'].append(actor_loss.item())
        self.training_error['critic_loss'].append(critic_loss.item())
        self.training_error['entropy'].append(entropy_bonus.item())


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

        # Additional data from action selection
        self.log_probs = []
        self.state_values = []
        self.entropies = []

    def add(self, state, action, reward, done, next_state):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))  # Convert bool to float
        self.next_states.append(next_state)

    def add_action_info(self, log_prob, state_value, entropy):
        """
        Add action selection info (called from select_action).
        This avoids recomputing these values during learning.
        """
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.entropies.append(entropy)

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
            np.array(self.log_probs),
            np.array(self.state_values),
            np.array(self.entropies)
        )

    def clear(self):
        """Clear the buffer after learning update."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.log_probs = []
        self.state_values = []
        self.entropies = []

    def __len__(self):
        return len(self.states)




