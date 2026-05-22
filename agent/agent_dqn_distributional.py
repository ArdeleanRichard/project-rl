import random

import numpy as np
import torch
import torch.optim as optim

from agent.agent_dqn import AgentDQN
from models.q_network import QNetworkDistributional


def _categorical_projection(rewards, dones, next_probs, atoms, v_min, v_max, n_atoms, gamma):
    """
    Categorical projection of the Bellman target (Algorithm 1 from C51 paper).

    For each next-distribution atom z_j, compute the Bellman-updated atom:
      T_z_j = clip(r + gamma*(1-done)*z_j, v_min, v_max)

    Then distribute the probability p_j onto the two neighbouring grid atoms
    l = floor(b) and u = ceil(b) where b = (T_z_j - v_min) / delta_z:
      m_l += p_j * (u - b)
      m_u += p_j * (b - l)

    BUG FIX: when l == u (atom lands exactly on a grid point), both
    coefficients (u-b) and (b-l) are zero, silently discarding the mass.
    We detect this case and assign the full probability to l.

    Args:
      rewards:    (B, 1)
      dones:      (B, 1)
      next_probs: (B, N) — target distribution for the greedy next action
      atoms:      (N,)   — fixed atom support [v_min, ..., v_max]
      gamma:      scalar discount (already raised to n-th power for n-step)

    Returns:
      m: (B, N) — projected target distribution (rows sum to 1)
    """
    batch_size = rewards.size(0)
    delta_z    = (v_max - v_min) / (n_atoms - 1)

    # Bellman-shifted atoms, clipped to [v_min, v_max]
    T_z = rewards + (1.0 - dones) * gamma * atoms.unsqueeze(0)  # (B, N)
    T_z = T_z.clamp(v_min, v_max)

    # Fractional atom indices
    b = (T_z - v_min) / delta_z          # (B, N)  in [0, N-1]
    l = b.floor().long().clamp(0, n_atoms - 1)
    u = b.ceil().long().clamp(0, n_atoms - 1)

    # Projection coefficients
    lower_frac = u.float() - b            # contribution to floor atom
    upper_frac = b - l.float()            # contribution to ceil atom

    # FIX: when l == u (b is exactly integer), both fracs are 0 — assign all mass to l
    eq_mask = (l == u)
    lower_frac[eq_mask] = 1.0
    upper_frac[eq_mask] = 0.0

    # Scatter probability mass onto the target distribution tensor
    m      = torch.zeros(batch_size, n_atoms, device=rewards.device)
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * n_atoms

    m.view(-1).scatter_add_(0, (l + offset).view(-1), (next_probs * lower_frac).view(-1))
    m.view(-1).scatter_add_(0, (u + offset).view(-1), (next_probs * upper_frac).view(-1))

    return m   # (B, N), each row sums to 1


class AgentDistributionalDQN(AgentDQN):
    """
    C51 — Categorical Distributional RL (Bellemare et al., 2017).

    Learns P(G_t | s, a) as a probability mass over N atoms in [v_min, v_max]
    instead of a scalar E[G_t | s, a].

    Action selection uses the expected Q: argmax_a sum_i z_i * p_i(s, a).
    Loss is cross-entropy between the projected Bellman target and the predicted
    distribution for the action taken.

    Three bugs fixed vs the previous version:
      1. v_min/v_max: set to [-200, 200] to cover actual LunarLander return range.
         Using [-10, 10] clips 100% of atoms for early-training episodes (~-200
         score), producing zero gradient signal.
      2. Categorical projection: when floor(b) == ceil(b) (atom lands exactly on
         a grid point), the previous code assigned 0 probability — all mass lost.
         Fixed by detecting the case and assigning full probability to that atom.
      3. Target action selection: use TARGET network (not local) for next-action
         selection in standalone C51. Using the local net (Double-DQN style) is
         correct for Rainbow where local and target are more decorrelated, but for
         vanilla C51 it adds instability without benefit.
    """

    def __init__(self, env, config, n_atoms=51, v_min=-200.0, v_max=200.0):
        super().__init__(env, config)

        self.n_atoms = n_atoms
        self.v_min   = v_min
        self.v_max   = v_max

        self.qnetwork_local  = QNetworkDistributional(
            self.n_states, self.n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        self.qnetwork_target = QNetworkDistributional(
            self.n_states, self.n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Atom support on self.device (also lives inside the networks as a buffer,
        # but we keep a reference here for use in learn())
        self.atoms = torch.linspace(v_min, v_max, n_atoms).to(self.device)

    def select_action(self, state, eps=0.0):
        """Epsilon-greedy over expected Q-values: Q(s,a) = sum_i z_i * p_i(s,a)."""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local.q_values(state)   # (1, action_size)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(q_values.cpu().argmax())
        return random.choice(range(self.n_actions))

    def learn(self, experiences):
        states, actions, rewards, dones, next_states = experiences
        batch_size = states.size(0)

        with torch.no_grad():
            # Use TARGET network for next-action selection (standard C51).
            # This is simpler and more stable than Double-DQN style for standalone C51.
            next_q       = self.qnetwork_target.q_values(next_states)   # (B, A)
            next_actions = next_q.argmax(dim=1)                          # (B,)

            next_probs = self.qnetwork_target.get_probs(next_states)     # (B, A, N)
            next_probs = next_probs[range(batch_size), next_actions]     # (B, N)

            # Project the Bellman target onto the atom grid
            m = _categorical_projection(
                rewards, dones, next_probs,
                self.atoms, self.v_min, self.v_max, self.n_atoms, self.GAMMA
            )

        # Predicted log-probabilities for the actions taken: (B, N)
        log_probs   = self.qnetwork_local(states)                        # (B, A, N)
        log_probs_a = log_probs[range(batch_size), actions.squeeze()]    # (B, N)

        # Cross-entropy loss: -sum_i m_i * log(p_i)
        loss = -(m * log_probs_a).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())