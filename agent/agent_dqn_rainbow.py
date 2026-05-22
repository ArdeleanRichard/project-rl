from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.agent_dqn_distributional import _categorical_projection, AgentDistributionalDQN
from agent.agent_dqn_prioritized import PrioritizedReplayBuffer
from models.q_network import QNetworkRainbow


class AgentRainbow(AgentDistributionalDQN):
    """
    Rainbow DQN (Hessel et al., 2017).

    Six improvements over vanilla DQN combined:
      1. Double Q-learning   — local net selects next action, target net evaluates it
      2. Prioritized replay  — high-TD-error transitions sampled more often
      3. Dueling networks    — separate V(s) and A(s,a) streams per atom
      4. Multi-step returns  — n-step Bellman backup, stored via MultiStepBuffer
      5. Distributional RL   — full return distribution (C51) with fixed atom range
      6. Noisy networks      — NoisyLinear replaces epsilon-greedy exploration

    Inheritance: AgentRainbow -> AgentDistributionalDQN -> AgentDQN
    We reuse the distributional act() (argmax expected Q) and override step/learn.
    """

    def __init__(self, env, config, n_atoms=51, v_min=-200.0, v_max=200.0, n_steps=3):
        super().__init__(env, config, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.n_steps = n_steps

        # Rainbow network: dueling + distributional + noisy
        self.qnetwork_local  = QNetworkRainbow(
            self.n_states, self.n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        self.qnetwork_target = QNetworkRainbow(
            self.n_states, self.n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Prioritized replay (improvement 2)
        self.memory = PrioritizedReplayBuffer(self.config)

        # Multi-step buffer (improvement 4)
        self.multistep = MultiStepBuffer(n_steps=n_steps, gamma=self.GAMMA)

    def select_action(self, state, eps=0.0):
        """
        Greedy w.r.t. expected Q-values. Noise provides exploration.
        Reset noise before each act so each action sees a fresh sample.
        """
        self.qnetwork_local.reset_noise()

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qnetwork_local.q_values(state)
        return int(q_values.cpu().argmax())

    def update(self, state, info, action, reward, done, next_state):
        """Pass raw transitions through the n-step buffer, then into PER."""
        completed = self.multistep.add(state, action, reward, done, next_state)
        for s, a, R, s_next, d in completed:
            self.memory.add(s, a, R, s_next, d)

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            experiences, indices, weights = self.memory.sample()
            self.learn(experiences, indices, weights)

    def learn(self, experiences, indices, weights):
        """
        Distributional Bellman update combining all six Rainbow improvements.

        gamma^n is used for the bootstrap (not gamma) because the stored rewards
        are already the n-step discounted sum R_n = sum_{k=0}^{n-1} gamma^k r_{t+k}.
        The correct Bellman target is: T_z = R_n + gamma^n * z_j.
        """
        states, actions, rewards, dones, next_states = experiences
        batch_size = states.size(0)

        # Resample noise in both networks (improvement 6)
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        with torch.no_grad():
            # Double DQN: local selects next action, target evaluates it (improvement 1)
            next_q       = self.qnetwork_local.q_values(next_states)       # (B, A)
            next_actions = next_q.argmax(dim=1)                             # (B,)

            next_probs = self.qnetwork_target.get_probs(next_states)        # (B, A, N)
            next_probs = next_probs[range(batch_size), next_actions]        # (B, N)

            # Categorical projection with gamma^n discount (improvements 4 + 5)
            gamma_n = self.GAMMA ** self.n_steps
            m = _categorical_projection(
                rewards, dones, next_probs,
                self.atoms, self.v_min, self.v_max, self.n_atoms, gamma_n
            )

        # Predicted log-probabilities for actions taken: (B, N)
        log_probs   = self.qnetwork_local(states)                           # (B, A, N)
        log_probs_a = log_probs[range(batch_size), actions.squeeze()]       # (B, N)

        # Per-sample cross-entropy loss
        elementwise_loss = -(m * log_probs_a).sum(dim=1)                    # (B,)

        # Weighted mean with IS weights (improvement 2)
        loss = (weights * elementwise_loss).mean()

        # Update priorities with per-sample loss as proxy for TD error
        self.memory.update_priorities(indices, elementwise_loss.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())



class MultiStepBuffer:
    """
    Accumulates raw transitions into n-step returns before storing in replay.

    For trajectory (s_t, a_t, r_t, ..., s_{t+n}):
      R_n = sum_{k=0}^{n-1} gamma^k * r_{t+k}

    Stores (s_t, a_t, R_n, s_{t+n}, done_{t+n}).
    The Bellman target for this transition is R_n + gamma^n * V(s_{t+n}),
    so the agent's learn() must use gamma^n (not gamma) for the bootstrap.
    """

    def __init__(self, n_steps, gamma):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.buffer  = deque()

    def add(self, state, action, reward, done, next_state):
        """
        Returns a list of completed n-step transitions (usually length 0 or 1).
        At episode end, returns all remaining transitions flushed from the buffer.
        """
        self.buffer.append((state, action, reward, done, next_state))

        if done:
            results = []
            while self.buffer:
                results.append(self._build())
                self.buffer.popleft()
            return results

        if len(self.buffer) == self.n_steps:
            result = self._build()
            self.buffer.popleft()
            return [result]

        return []

    def _build(self):
        """Build an n-step return from the front of the buffer."""
        buf = list(self.buffer)
        R   = 0.0
        for k, (_, _, r, _, d) in enumerate(buf):
            R += (self.gamma ** k) * r
            if d:
                # Episode ended at step k — no bootstrap beyond here
                return (buf[0][0], buf[0][1], R, buf[k][3], True)
        return (buf[0][0], buf[0][1], R, buf[-1][3], buf[-1][4])