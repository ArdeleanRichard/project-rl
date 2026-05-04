import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, QNetworkDueling, QNetworkDistributional, QNetworkNoisy, QNetworkRainbow

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE  = int(1e5)   # replay buffer size
BATCH_SIZE   = 64         # minibatch size
GAMMA        = 0.99       # discount factor
TAU          = 1e-3       # for soft update of target parameters
LR           = 5e-4       # learning rate
UPDATE_EVERY = 4          # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffers
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Uniform random experience replay buffer."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory      = deque(maxlen=buffer_size)
        self.batch_size  = batch_size
        self.experience  = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states      = torch.from_numpy(np.vstack([e.state      for e in experiences])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (Schaul et al., 2015).

    Samples transitions proportional to their TD error:
      P(i) = p_i^alpha / sum_k p_k^alpha
    Importance-sampling weights correct the resulting bias:
      w_i = (N * P(i))^{-beta}
    New transitions receive max priority so they are sampled at least once.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        super().__init__(action_size, buffer_size, batch_size, seed)
        self.alpha       = alpha
        self.pos         = 0
        self.buffer_size = buffer_size
        self.priorities  = np.zeros((buffer_size,), dtype=np.float32)
        self.memory      = []   # list for O(1) index access

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, beta=0.4):
        N     = len(self.memory)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # replace=True is the paper standard; avoids edge cases with small buffers
        indices     = np.random.choice(N, self.batch_size, replace=True, p=probs)
        experiences = [self.memory[i] for i in indices]

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights.astype(np.float32)).to(device)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(float(err)) + 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Base Agent — Vanilla DQN
# ──────────────────────────────────────────────────────────────────────────────

class AgentDQN:
    """Vanilla DQN (Mnih et al., 2015)."""

    def __init__(self, state_size, action_size, seed):
        self.state_size  = state_size
        self.action_size = action_size
        self.seed        = random.seed(seed)

        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            self.learn(self.memory.sample(), GAMMA)

    def act(self, state, eps=0.0):
        """Epsilon-greedy action selection."""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(action_values.cpu().argmax())
        return random.choice(range(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """theta_target = tau*theta_local + (1-tau)*theta_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# ──────────────────────────────────────────────────────────────────────────────
# Double DQN
# ──────────────────────────────────────────────────────────────────────────────

class AgentDoubleDQN(AgentDQN):
    """
    Double DQN (van Hasselt et al., 2015).

    Standard DQN uses the target net to both select AND evaluate the best next
    action, causing systematic overestimation. Fix: local net selects the action,
    target net evaluates it.
    """

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        best_actions   = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        Q_targets      = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ──────────────────────────────────────────────────────────────────────────────
# Prioritized DQN
# ──────────────────────────────────────────────────────────────────────────────

class AgentPriorityDQN(AgentDQN):
    """
    Prioritized Experience Replay DQN (Schaul et al., 2015).

    Samples high-TD-error transitions more often. IS weights correct the bias.
    """

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences, indices, weights = self.memory.sample()
            self.learn(experiences, GAMMA, indices, weights)

    def learn(self, experiences, gamma, indices, weights):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        td_errors = (Q_expected - Q_targets).detach().squeeze().cpu().numpy()
        loss      = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none').squeeze()).mean()
        self.memory.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ──────────────────────────────────────────────────────────────────────────────
# Dueling DQN
# ──────────────────────────────────────────────────────────────────────────────

class AgentDuelingDQN(AgentDQN):
    """
    Dueling Network DQN (Wang et al., 2016).

    Separate V(s) and A(s,a) streams. Q = V + (A - mean(A)).
    learn() is identical to vanilla DQN — only the architecture changes.
    """

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        self.qnetwork_local  = QNetworkDueling(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetworkDueling(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


# ──────────────────────────────────────────────────────────────────────────────
# Distributional DQN  (C51)
# ──────────────────────────────────────────────────────────────────────────────

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

    def __init__(self, state_size, action_size, seed,
                 n_atoms=51, v_min=-200.0, v_max=200.0):
        super().__init__(state_size, action_size, seed)

        self.n_atoms = n_atoms
        self.v_min   = v_min
        self.v_max   = v_max

        self.qnetwork_local  = QNetworkDistributional(
            state_size, action_size, seed, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.qnetwork_target = QNetworkDistributional(
            state_size, action_size, seed, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Atom support on device (also lives inside the networks as a buffer,
        # but we keep a reference here for use in learn())
        self.atoms = torch.linspace(v_min, v_max, n_atoms).to(device)

    def act(self, state, eps=0.0):
        """Epsilon-greedy over expected Q-values: Q(s,a) = sum_i z_i * p_i(s,a)."""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local.q_values(state)   # (1, action_size)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(q_values.cpu().argmax())
        return random.choice(range(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
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
                self.atoms, self.v_min, self.v_max, self.n_atoms, gamma
            )

        # Predicted log-probabilities for the actions taken: (B, N)
        log_probs   = self.qnetwork_local(states)                        # (B, A, N)
        log_probs_a = log_probs[range(batch_size), actions.squeeze()]    # (B, N)

        # Cross-entropy loss: -sum_i m_i * log(p_i)
        loss = -(m * log_probs_a).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ──────────────────────────────────────────────────────────────────────────────
# Noisy DQN
# ──────────────────────────────────────────────────────────────────────────────

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

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        self.qnetwork_local  = QNetworkNoisy(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetworkNoisy(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    def act(self, state, eps=0.0):
        """
        Greedy w.r.t. the noisy network. Noise provides the exploration.
        reset_noise() is called here so each action sees a fresh noise sample.
        The network stays in train() mode so NoisyLinear uses noise.
        """
        # Resample noise before each action for diverse exploration
        self.qnetwork_local.reset_noise()

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return int(action_values.cpu().argmax())

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Resample noise in both networks so each learning step sees a
        # fresh realisation and the target is not biased by stale noise.
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step Buffer  (used by Rainbow)
# ──────────────────────────────────────────────────────────────────────────────

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

    def add(self, state, action, reward, next_state, done):
        """
        Returns a list of completed n-step transitions (usually length 0 or 1).
        At episode end, returns all remaining transitions flushed from the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

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


# ──────────────────────────────────────────────────────────────────────────────
# Rainbow Agent
# ──────────────────────────────────────────────────────────────────────────────

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

    def __init__(self, state_size, action_size, seed,
                 n_atoms=51, v_min=-200.0, v_max=200.0, n_steps=3):
        super().__init__(state_size, action_size, seed,
                         n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.n_steps = n_steps

        # Rainbow network: dueling + distributional + noisy
        self.qnetwork_local  = QNetworkRainbow(
            state_size, action_size, seed, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.qnetwork_target = QNetworkRainbow(
            state_size, action_size, seed, n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Prioritized replay (improvement 2)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Multi-step buffer (improvement 4)
        self.multistep = MultiStepBuffer(n_steps=n_steps, gamma=GAMMA)

    def act(self, state, eps=0.0):
        """
        Greedy w.r.t. expected Q-values. Noise provides exploration.
        Reset noise before each act so each action sees a fresh sample.
        """
        self.qnetwork_local.reset_noise()

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.qnetwork_local.q_values(state)
        return int(q_values.cpu().argmax())

    def step(self, state, action, reward, next_state, done):
        """Pass raw transitions through the n-step buffer, then into PER."""
        completed = self.multistep.add(state, action, reward, next_state, done)
        for s, a, R, s_next, d in completed:
            self.memory.add(s, a, R, s_next, d)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences, indices, weights = self.memory.sample()
            self.learn(experiences, GAMMA, indices, weights)

    def learn(self, experiences, gamma, indices, weights):
        """
        Distributional Bellman update combining all six Rainbow improvements.

        gamma^n is used for the bootstrap (not gamma) because the stored rewards
        are already the n-step discounted sum R_n = sum_{k=0}^{n-1} gamma^k r_{t+k}.
        The correct Bellman target is: T_z = R_n + gamma^n * z_j.
        """
        states, actions, rewards, next_states, dones = experiences
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
            gamma_n = gamma ** self.n_steps
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
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    STATE, ACTION, SEED = 8, 4, 0

    agents = [
        (AgentDQN,               "Vanilla DQN"),
        (AgentDoubleDQN,         "Double DQN"),
        (AgentPriorityDQN,       "Priority DQN"),
        (AgentDuelingDQN,        "Dueling DQN"),
        (AgentDistributionalDQN, "Distributional DQN"),
        (AgentNoisyDQN,          "Noisy DQN"),
    ]
    for AgentCls, name in agents:
        a = AgentCls(STATE, ACTION, SEED)
        n = sum(p.numel() for p in a.qnetwork_local.parameters())
        print(f"{name:<25}: {n:>7} params")

    r = AgentRainbow(STATE, ACTION, SEED, n_steps=3)
    n = sum(p.numel() for p in r.qnetwork_local.parameters())
    print(f"{'Rainbow':<25}: {n:>7} params")

    # Verify projection sums to 1 for all edge cases
    atoms = torch.linspace(-200, 200, 51)
    rewards    = torch.tensor([[-150.0], [0.0], [200.0]])
    dones      = torch.tensor([[0.0], [0.0], [1.0]])
    next_probs = torch.ones(3, 51) / 51
    m = _categorical_projection(rewards, dones, next_probs, atoms, -200, 200, 51, 0.99)
    assert torch.allclose(m.sum(dim=1), torch.ones(3), atol=1e-5), \
        f"Projection rows don't sum to 1: {m.sum(dim=1)}"
    print("\nCategorical projection row sums:", m.sum(dim=1).tolist(), "✓")

    # Verify noise varies between resets
    ag = AgentNoisyDQN(STATE, ACTION, SEED)
    s  = torch.randn(1, STATE).to(device)
    ag.qnetwork_local.reset_noise()
    q1 = ag.qnetwork_local(s).detach()
    ag.qnetwork_local.reset_noise()
    q2 = ag.qnetwork_local(s).detach()
    assert not torch.allclose(q1, q2), "Noise not varying between resets!"
    print("NoisyDQN noise varies between resets ✓")