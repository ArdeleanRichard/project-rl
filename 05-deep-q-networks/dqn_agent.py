import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, QNetworkDueling, QNetworkDistributional, QNetworkNoisy, QNetworkRainbow

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDQN():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # - exploitation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        # - exploration
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        - Use MSE(target, expected)

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model -  - gather on dim 1, all values given by actions
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class AgentDoubleDQN(AgentDQN):
    def __init__(self, state_size, action_size, seed):
        """
        Standard DQN uses the target network to both select and evaluate the best next action. 
        - This causes systematic overestimation of Q-values — the agent becomes overconfident.

        Double DQN MODIFICATION:
        - Use the local network to select the best action, and the target network to evaluate it. Two separate networks → unbiased estimates.

        Why?
        - Standard: max(target(s')) — same network selects AND scores the action
        - Double: target(s')[argmax(local(s'))] — decorrelated selection vs evaluation
        """
        super().__init__(state_size, action_size, seed)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        - Use MSE(target, expected)

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Standard DQN uses the target network to both select and evaluate the best next action. 
        # - This causes systematic overestimation of Q-values — the agent becomes overconfident.
        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Double DQN MODIFICATION:
        # Use the local network to select the best action, and the target network to evaluate it. Two separate networks → unbiased estimates.
        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model -  - gather on dim 1, all values given by actions
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)           


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




class AgentPriorityDQN(AgentDQN):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """
        ReplayBuffer samples uniformly at random. Rare but important transitions (big surprises) are seen just as often as boring ones.

        Sample transitions in proportion to their TD error (— )how wrong the network was). Add importance-sampling weights to correct the bias this introduces.


        """
        super().__init__(state_size, action_size, seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, weights = self.memory.sample()
                self.learn(experiences, GAMMA, indices, weights)


    def learn(self, experiences, gamma, indices, weights):
        """Update value parameters using given batch of experience tuples.
        - Use MSE(target, expected)

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model -  - gather on dim 1, all values given by actions
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        td_errors = (Q_expected - Q_targets).detach().squeeze().cpu().numpy()
        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none').squeeze()).mean()
        self.memory.update_priorities(indices, td_errors)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     



class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.5):
        super().__init__(action_size, buffer_size, batch_size, seed)

        self.alpha = alpha        # how much to prioritize (0=uniform, 1=full)
        self.pos = 0

        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.memory = []
        self.buffer_size = buffer_size


    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0

        e = self.experience(state, action, reward, next_state, done)

        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        self.priorities[self.pos] = max_prio  # new transitions get max priority
        self.pos = (self.pos + 1) % self.buffer_size


    def sample(self, beta=0.4):
        N = len(self.memory)

        prios = self.priorities[:N]
        probs = prios ** self.alpha / (prios ** self.alpha).sum()
        indices = np.random.choice(N, self.batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]

        # Importance-sampling weights to fix the sampling bias
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), indices, weights


    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + 1e-5  # small epsilon avoids zero priority



class AgentDuelingDQN(AgentDQN):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        """Initialize an Agent object.
        
        The problem
        - A single stream learns Q(s,a) for each action. In many states the choice of action barely matters 
        - the network wastes capacity learning action differences when the state value is what counts.
        The fix
        - Split the network into two streams: Value V(s) — how good is this state? and Advantage A(s,a) — how much better is action a vs the average? 
        - Then recombine: Q = V + (A − mean(A)).
        Why subtract mean(A)?
        - Without it, V and A are not uniquely recoverable from Q — the decomposition is unidentifiable. 
        - Subtracting the mean forces the advantage to sum to zero, making V and A each carry distinct meaning.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        # Q-DuelingNetwork 
        self.qnetwork_local = QNetworkDueling(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetworkDueling(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)




class AgentDistributionalDQN(AgentDQN):
    """
    The problem:
    - Standard DQN predicts E[return] — a single scalar per action.
    - The mean alone loses all information about risk, variance, multi-modal returns.
 
    The fix — Categorical / C51:
    - Represent the return distribution as a categorical distribution over
        N_ATOMS fixed support points z ∈ [V_MIN, V_MAX].
    - The network outputs a probability vector p(s,a) of length n_atoms for
        each action.  Q(s,a) = Σ z_i · p_i  is then derived, not directly predicted.
    - Training minimises KL(projected target distribution ‖ predicted distribution).
 
    Why does projection matter?
    - After a Bellman update r + γ·z the atoms shift and no longer lie on the
        fixed support grid.  We must "project" the shifted atoms back onto the
        grid before computing the KL loss — this is the distributional Bellman op.
    """
 
    # Distributional hyperparameters
    N_ATOMS = 51
    V_MIN   = -10.0
    V_MAX   =  10.0
 
    def __init__(self, state_size, action_size, seed):
        # Call grandparent AgentDQN.__init__ to get memory, t_step, etc.
        super().__init__(state_size, action_size, seed)
 
        # Replace scalar Q-networks with distributional ones
        self.qnetwork_local  = QNetworkDistributional(
            state_size, action_size, seed,
            n_atoms=self.N_ATOMS, v_min=self.V_MIN, v_max=self.V_MAX
        ).to(device)
        self.qnetwork_target = QNetworkDistributional(
            state_size, action_size, seed,
            n_atoms=self.N_ATOMS, v_min=self.V_MIN, v_max=self.V_MAX
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
 
        # Precompute the fixed support atoms on the right device
        self.atoms = torch.linspace(self.V_MIN, self.V_MAX, self.N_ATOMS).to(device)
        self.delta_z = (self.V_MAX - self.V_MIN) / (self.N_ATOMS - 1)
 
    # ── act: select action by expected Q = Σ z·p ─────────────────────────────
    def act(self, state, eps=0.):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            # q_values() returns scalar Q per action — same interface as base class
            action_values = self.qnetwork_local.q_values(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))
 
    # ── learn: distributional Bellman update ──────────────────────────────────
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.shape[0]
 
        with torch.no_grad():
            # Select greedy actions in next states using the *local* network
            #    (Double-DQN style — reduces overestimation)
            next_q     = self.qnetwork_local.q_values(next_states)   # (B, A)
            next_acts  = next_q.argmax(1)                             # (B,)
 
            # Get the target distribution for those greedy actions
            next_probs = self.qnetwork_target(next_states)            # (B, A, N)
            next_probs = next_probs[range(batch_size), next_acts]     # (B, N)
 
            # Project the Bellman-updated atoms back onto the fixed support
            #    Tz_j = clip(r + γ·z_j, V_MIN, V_MAX)
            Tz = rewards + (1 - dones) * gamma * self.atoms.unsqueeze(0)  # (B, N)
            Tz = Tz.clamp(self.V_MIN, self.V_MAX)
 
            # Compute lower/upper atom indices and their interpolation coefficients
            b  = (Tz - self.V_MIN) / self.delta_z          # (B, N) float index
            lo = b.floor().long().clamp(0, self.N_ATOMS - 1)
            hi = b.ceil().long().clamp(0, self.N_ATOMS - 1)
 
            # Distribute probability mass proportionally between lo and hi
            m = torch.zeros(batch_size, self.N_ATOMS, device=device)
            offset = torch.arange(batch_size, device=device).unsqueeze(1) * self.N_ATOMS
            m.view(-1).scatter_add_(0, (lo + offset).view(-1), (next_probs * (hi.float() - b)).view(-1))
            m.view(-1).scatter_add_(0, (hi + offset).view(-1), (next_probs * (b - lo.float())).view(-1))
            # m is now the target distribution (B, N) on the fixed support
 
        # 4. Get predicted log-probabilities for the taken actions
        log_probs = torch.log(self.qnetwork_local(states)[range(batch_size), actions.squeeze()] + 1e-8)  # (B, N)
 
        # Cross-entropy loss = -Σ m · log p  (equivalent to KL up to a constant)
        loss = -(m * log_probs).sum(dim=1).mean()
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
 
 

 
class AgentNoisyDQN(AgentDQN):
    """
    The problem:
    - ε-greedy exploration is undirected: the same random noise is applied
      regardless of the state or how uncertain the agent actually is.
 
    The fix — NoisyNets:
    - Replace all Linear layers with NoisyLinear layers (μ + σ·ε weights).
    - The network learns *per-weight* noise scales σ — naturally exploring more
      in states where it is uncertain and less where it is confident.
    - No external ε schedule is needed; pass eps=0 to act() always.
 
    The only code change vs AgentDQN:
    - Swap QNetwork → QNetworkNoisy.
    - Call reset_noise() after every gradient step so the next forward pass
      uses fresh noise samples.
    """
 
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
 
        # Replace scalar Q-networks with noisy versions
        self.qnetwork_local  = QNetworkNoisy(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetworkNoisy(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
 

    def act(self, state, eps=0.):
        """
        No ε-greedy needed — the noisy network provides built-in exploration.
        eps is kept in the signature for compatibility with the dqn() training
        loop but is ignored (equivalent to always exploiting, with noise
        doing the exploration internally).
        """
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Always greedy — noise in the network handles exploration
        return np.argmax(action_values.cpu().data.numpy())
 

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
 
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets  = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
 
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # Re-sample noise so the next forward pass explores differently
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
 
 
class MultiStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Extends PrioritizedReplayBuffer with n-step return accumulation.
 
    The problem with 1-step TD:
    - Each update propagates reward information by only one step.
    - Learning is slow because early states in a trajectory must wait many
      updates before they "feel" a reward that happened several steps later.
 
    The fix — n-step returns:
    - Accumulate rewards over n steps:  G_t = r_t + γ·r_{t+1} + … + γ^{n-1}·r_{t+n-1}
    - Use s_{t+n} as the bootstrap state.
    - This effectively skips n-1 intermediate Bellman backups, making credit
      assignment faster at the cost of a slight bias when n > 1.
 
    Implementation:
    - A small deque (n_step_buffer) holds the last n transitions.
    - Once full, we compute the accumulated return and store the compressed
      (s_t, a_t, G_t, s_{t+n}, done_{t+n}) transition into the priority buffer.
    """
 
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta_start=0.4, beta_frames=100_000, n_steps=3, gamma=GAMMA):
        super().__init__(action_size, buffer_size, batch_size, seed, alpha)
        self.n_steps      = n_steps
        self.gamma        = gamma
        # Temporary queue that collects the last n raw transitions
        self.n_step_buffer = deque(maxlen=n_steps)
 

    def add(self, state, action, reward, next_state, done):
        """Buffer a transition; only store to PER once n steps are accumulated."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
 
        if len(self.n_step_buffer) < self.n_steps:
            # Not enough steps yet — keep collecting
            return
 
        # Compute the n-step discounted return G_t
        G          = 0.0
        final_done = False
        final_next = self.n_step_buffer[-1][3]   # s_{t+n}
        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            G += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps — use the terminal state
                final_done = True
                final_next = ns
                break
 
        # Store the compressed transition into the prioritized buffer
        first = self.n_step_buffer[0]
        super().add(first[0], first[1], G, final_next, final_done)
 
 
 
class AgentRainbow(AgentDQN):
    """
    Rainbow combines six DQN improvements:
        1. Double DQN          — local net selects action, target net evaluates it
        2. Prioritized Replay  — sample by |TD error|, correct bias with IS weights
        3. Dueling Networks    — separate value and advantage streams
        4. Multi-step returns  — n-step TD targets for faster credit assignment
        5. Distributional RL   — predict full return distribution (C51)
        6. Noisy Nets          — learned per-weight exploration noise
 
    Architecture (3 + 5 + 6) lives in QNetworkRainbow.
    Replay (2 + 4) lives in MultiStepPrioritizedReplayBuffer.
    Learning (1) is implemented in learn() below.
 
    act() passes eps=0 — exploration is handled entirely by NoisyLinear layers.
    """
 
    # Distributional hyperparameters (same as AgentDistributionalDQN for consistency)
    N_ATOMS = 51
    V_MIN   = -10.0
    V_MAX   =  10.0
    N_STEPS      = 3        # multi-step return horizon (used by Rainbow)
 
    def __init__(self, state_size, action_size, seed, n_steps=N_STEPS):
        super().__init__(state_size, action_size, seed)
 
        # ── Architecture: Dueling + Distributional + Noisy ──────────────────
        self.qnetwork_local  = QNetworkRainbow(
            state_size, action_size, seed,
            n_atoms=self.N_ATOMS, v_min=self.V_MIN, v_max=self.V_MAX
        ).to(device)
        self.qnetwork_target = QNetworkRainbow(
            state_size, action_size, seed,
            n_atoms=self.N_ATOMS, v_min=self.V_MIN, v_max=self.V_MAX
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
 
        # ── Replay: Prioritized + Multi-step ─────────────────────────────────
        self.memory = MultiStepPrioritizedReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, seed,
            n_steps=n_steps, gamma=GAMMA
        )
        self.n_steps = n_steps
 
        # Support atoms
        self.atoms   = torch.linspace(self.V_MIN, self.V_MAX, self.N_ATOMS).to(device)
        self.delta_z = (self.V_MAX - self.V_MIN) / (self.N_ATOMS - 1)
 

    # ── act: greedy over expected Q — noise handles exploration ───────────────
    def act(self, state, eps=0.):
        """eps is ignored; NoisyLinear layers provide intrinsic exploration."""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.q_values(state)
        self.qnetwork_local.train()
        return np.argmax(action_values.cpu().data.numpy())
 

    # ── step: pull (experiences, indices, weights) from the PER buffer ────────
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences, indices, weights = self.memory.sample()
            self.learn(experiences, GAMMA, indices, weights)
 

    # ── learn: distributional Bellman + IS weights + Double DQN ───────────────
    def learn(self, experiences, gamma, indices, weights):
        """
        Combines:
        - Double DQN (1): local selects next action, target evaluates
        - Prioritized replay (2): weighted loss + priority update
        - Distributional (5): KL loss on projected target distribution
        - Multi-step (4): gamma is γ^n since rewards are already n-step sums
        - Noise (6): reset_noise() called after the update
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.shape[0]
 
        # γ^n for the bootstrap — rewards already contain the n-step sum
        gamma_n = gamma ** self.n_steps
 
        with torch.no_grad():
            # 1. Double DQN: local picks next action, target evaluates distribution
            next_q    = self.qnetwork_local.q_values(next_states)     # (B, A)
            next_acts = next_q.argmax(1)                               # (B,)
            next_probs = self.qnetwork_target(next_states)             # (B, A, N)
            next_probs = next_probs[range(batch_size), next_acts]      # (B, N)
 
            # 2. Project Bellman-updated atoms onto fixed support
            #    Tz_j = clip(r_n + γ^n · z_j, V_MIN, V_MAX)
            Tz = rewards + (1 - dones) * gamma_n * self.atoms.unsqueeze(0)  # (B, N)
            Tz = Tz.clamp(self.V_MIN, self.V_MAX)
 
            b  = (Tz - self.V_MIN) / self.delta_z
            lo = b.floor().long().clamp(0, self.N_ATOMS - 1)
            hi = b.ceil().long().clamp(0, self.N_ATOMS - 1)
 
            m = torch.zeros(batch_size, self.N_ATOMS, device=device)
            offset = torch.arange(batch_size, device=device).unsqueeze(1) * self.N_ATOMS
            m.view(-1).scatter_add_(0, (lo + offset).view(-1),
                                    (next_probs * (hi.float() - b)).view(-1))
            m.view(-1).scatter_add_(0, (hi + offset).view(-1),
                                    (next_probs * (b - lo.float())).view(-1))
 
        # 3. Predicted log-probabilities for the taken actions
        log_probs = torch.log(
            self.qnetwork_local(states)[range(batch_size), actions.squeeze()] + 1e-8
        )  # (B, N)
 
        # 4. Weighted cross-entropy loss (IS weights from PER)
        elementwise_loss = -(m * log_probs).sum(dim=1)          # (B,)
        loss = (weights * elementwise_loss).mean()
 
        # 5. Update priorities with the new TD errors
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # 6. Re-sample noise for the next step
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)







if __name__ == "__main__":
    agent = AgentDQN(state_size=8, action_size=4, seed=0)
    print(len(list(agent.qnetwork_local.parameters())))