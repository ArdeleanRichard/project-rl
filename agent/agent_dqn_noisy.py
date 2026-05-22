import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.agent_dqn import AgentDQN
from models.q_network import QNetworkNoisy


class AgentNoisyDQN(AgentDQN):
    """
    Noisy DQN (Fortunato et al., 2017).

    Replaces epsilon-greedy with parametric noise in the network weights.
    The noise magnitude sigma is learnable: the network decides how much
    randomness is useful in each state, annealing toward zero as it gains
    confidence.

    Two fixes vs the previous version:
      1. reset_noise() is now called once per ACT step (not just learn), so
         each action selection uses a freshly sampled noise realisation. The
         previous version reused the same noise for the entire episode rollout,
         severely limiting exploration diversity.
      2. reset_noise() is also called on BOTH local and target networks in
         learn(), so targets are not evaluated with stale noise.

    Usage: pass eps_start=0, eps_end=0, eps_decay=1.0 to the training loop.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.qnetwork_local  = QNetworkNoisy(self.n_states, self.n_actions, self.seed).to(self.device)
        self.qnetwork_target = QNetworkNoisy(self.n_states, self.n_actions, self.seed).to(self.device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

    def select_action(self, state, eps=0.0):
        """
        Greedy w.r.t. the noisy network. Noise provides the exploration.
        reset_noise() is called here so each action sees a fresh noise sample.
        The network stays in train() mode so NoisyLinear uses noise.
        """
        # Resample noise before each action for diverse exploration
        self.qnetwork_local.reset_noise()

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return int(action_values.cpu().argmax())

    def learn(self, experiences):
        states, actions, rewards, dones, next_states = experiences

        # Resample noise in both networks so each learning step sees a
        # fresh realisation and the target is not biased by stale noise.
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + self.GAMMA * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())
