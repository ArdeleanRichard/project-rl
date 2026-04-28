import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers=[64,64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # TODO: implement
        layer_sizes = [state_size] + layers + [action_size]
        
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x)


class QNetworkDueling(nn.Module):

    def __init__(self, state_size, action_size, seed, layers=[64,64]):
        """
        The problem
        - A single stream learns Q(s,a) for each action. In many states the choice of action barely matters 
        - the network wastes capacity learning action differences when the state value is what counts.
        The fix
        - Split the network into two streams: Value V(s) — how good is this state? and Advantage A(s,a) — how much better is action a vs the average? 
        - Then recombine: Q = V + (A − mean(A)).
        Why subtract mean(A)?
        - Without it, V and A are not uniquely recoverable from Q — the decomposition is unidentifiable. 
        - Subtracting the mean forces the advantage to sum to zero, making V and A each carry distinct meaning.
        """
        super(QNetworkDueling, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Shared feature layers
        shared = [state_size] + layers
        self.shared = nn.ModuleList([
            nn.Linear(shared[i], shared[i+1])
            for i in range(len(shared) - 1)
        ])

        hidden = layers[-1]

        # Value stream: hidden → 1 scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Advantage stream: hidden → action_size
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )



    def forward(self, state):
        x = state
        for fc in self.shared:
            x = F.relu(fc(x))

        V = self.value_stream(x)          # shape: (batch, 1)
        A = self.advantage_stream(x)      # shape: (batch, action_size)

        # Combine: subtract mean advantage for identifiability
        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q
    



# ──────────────────────────────────────────────────────────────────────────────
# NoisyLinear — shared building block for Noisy DQN and Rainbow
# ──────────────────────────────────────────────────────────────────────────────
 
class NoisyLinear(nn.Module):
    """
    A linear layer whose weights and biases include a learned noise component.
 
    The problem with ε-greedy exploration:
    - ε-greedy adds the same undirected noise to every action, in every state.
    - It cannot focus exploration where the agent is actually uncertain.
 
    The fix — NoisyNets (Fortunato et al. 2017):
    - Replace each weight w with  w = μ + σ · ε,  where μ and σ are learned
      parameters and ε is random noise sampled each forward pass.
    - The network learns *how much* noise is useful per weight → state-dependent,
      action-dependent exploration for free, no ε schedule needed.
 
    We use factorised Gaussian noise (cheaper than independent noise):
    - Sample p noise values for inputs and q for outputs (p+q instead of p·q).
    - Combine them as outer product to get the full weight-noise matrix.
    """
 
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
 
        # Learnable mean parameters (same shape as a normal Linear layer)
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
 
        # Noise buffers — not parameters, re-sampled every forward pass
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))
 
        self.sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()
 
    def _reset_parameters(self):
        """Initialise μ with uniform ±1/√fan_in, σ with a small constant."""
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        # σ_init / √fan_in is the recommended initialisation from the paper
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
 
    @staticmethod
    def _f(x):
        """Factorised noise transform: sgn(x) · √|x|."""
        return x.sign() * x.abs().sqrt()
 
    def reset_noise(self):
        """Re-sample factorised Gaussian noise (call once per learning step)."""
        eps_in  = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        # Outer product gives the full (out, in) noise matrix cheaply
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)
 
    def forward(self, x):
        if self.training:
            # Perturbed weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # At evaluation time use only the mean (deterministic)
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Distributional Q-Network  (C51 — Bellemare et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────
 
class QNetworkDistributional(nn.Module):
    """
    The problem with scalar Q-values:
    - Q(s,a) = E[return] collapses the full distribution of possible returns
      to a single number.  Two actions with identical means but very different
      risk profiles look the same to the agent.
 
    The fix — Distributional RL / C51:
    - Instead of predicting E[return], predict the full *distribution* of returns
      as a categorical distribution over N_ATOMS fixed support points
      [V_MIN, …, V_MAX].
    - The network outputs a (batch, action_size, n_atoms) logit tensor.
    - After softmax each row is a probability distribution over the atoms.
    - Q(s,a) = Σ_i  z_i · p_i(s,a)   (expected value, for action selection).
 
    Why does this help?
    - Richer learning signal — the KL-divergence loss trains the shape of the
      distribution, not just its mean.
    - The agent naturally represents uncertainty and multi-modal returns.
    """
 
    def __init__(self, state_size, action_size, seed,
                 layers=[64, 64], n_atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.seed      = torch.manual_seed(seed)
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max
        # The fixed support atoms z_0 … z_{N-1}
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
 
        # Shared feature trunk (reuses the same pattern as QNetwork)
        layer_sizes = [state_size] + layers
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
 
        # Output head: for each action output a logit per atom
        self.out = nn.Linear(layers[-1], action_size * n_atoms)
 
    def forward(self, state):
        """
        Returns:
            probs  — shape (batch, action_size, n_atoms), softmax probabilities
        """
        x = state
        for fc in self.fcs:
            x = F.relu(fc(x))
 
        # Reshape to (batch, action_size, n_atoms) then softmax over atoms
        logits = self.out(x).view(-1, self.out.out_features // self.n_atoms, self.n_atoms)
        return F.softmax(logits, dim=2)   # probabilities over atoms
 
    def q_values(self, state):
        """Scalar Q(s,a) = Σ z_i · p_i(s,a) — used for action selection."""
        probs = self.forward(state)                  # (batch, action_size, n_atoms)
        return (probs * self.atoms).sum(dim=2)        # (batch, action_size)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Noisy Q-Network  (Fortunato et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────
 
class QNetworkNoisy(nn.Module):
    """
    Identical architecture to QNetwork but all Linear layers are replaced with
    NoisyLinear layers, giving the network intrinsic, state-dependent exploration.
 
    Because the noise is learned, no external ε-greedy schedule is needed —
    the agent explores more in states it is uncertain about and exploits where
    it is confident.
    """
 
    def __init__(self, state_size, action_size, seed, layers=[64, 64], sigma_init=0.5):
        super().__init__()
        self.seed = torch.manual_seed(seed)
 
        # NoisyLinear replaces every nn.Linear in the trunk
        layer_sizes = [state_size] + layers + [action_size]
        self.fcs = nn.ModuleList([
            NoisyLinear(layer_sizes[i], layer_sizes[i + 1], sigma_init)
            for i in range(len(layer_sizes) - 1)
        ])
 
    def forward(self, state):
        x = state
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x)
 
    def reset_noise(self):
        """Re-sample noise in all NoisyLinear layers (call once per learn step)."""
        for fc in self.fcs:
            fc.reset_noise()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Rainbow Q-Network  (Hessel et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────
 
class QNetworkRainbow(nn.Module):
    """
    Rainbow combines six improvements to DQN into one agent:
        1. Double DQN          — decorrelated action selection & evaluation
        2. Prioritized Replay  — sample important transitions more often
        3. Dueling Networks    — separate value and advantage streams
        4. Multi-step returns  — n-step TD targets (handled in the agent)
        5. Distributional RL   — predict full return distribution (C51)
        6. Noisy Nets          — learned, state-dependent exploration
 
    This network implements improvements 3, 5, and 6 at the architecture level:
    - NoisyLinear in every layer (6)
    - Dueling streams for value V and advantage A (3)
    - Each stream outputs n_atoms logits, not a scalar (5)
 
    Improvements 1, 2, and 4 are handled by AgentRainbow / PrioritizedReplayBuffer.
    """
 
    def __init__(self, state_size, action_size, seed,
                 layers=[64, 64], n_atoms=51, v_min=-10.0, v_max=10.0,
                 sigma_init=0.5):
        super().__init__()
        self.seed      = torch.manual_seed(seed)
        self.n_atoms   = n_atoms
        self.v_min     = v_min
        self.v_max     = v_max
        self.action_size = action_size
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
 
        # ── Shared noisy trunk ──────────────────────────────────────────────
        layer_sizes = [state_size] + layers
        self.shared = nn.ModuleList([
            NoisyLinear(layer_sizes[i], layer_sizes[i + 1], sigma_init)
            for i in range(len(layer_sizes) - 1)
        ])
 
        hidden = layers[-1]
 
        # ── Value stream (dueling): hidden → 1 × n_atoms ───────────────────
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden, n_atoms, sigma_init)          # V distribution
        )
 
        # ── Advantage stream (dueling): hidden → action_size × n_atoms ─────
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden, action_size * n_atoms, sigma_init)  # A distribution
        )
 
    def forward(self, state):
        """
        Returns:
            probs — shape (batch, action_size, n_atoms), softmax probabilities.
 
        Combines dueling (V+A) and distributional (softmax over atoms):
            Q_logits = V + (A - mean(A))   [per atom]
            Q_probs  = softmax(Q_logits, dim=atoms)
        """
        x = state
        for fc in self.shared:
            x = F.relu(fc(x))
 
        # Value: (batch, 1, n_atoms)
        V = self.value_stream(x).view(-1, 1, self.n_atoms)
 
        # Advantage: (batch, action_size, n_atoms)
        A = self.advantage_stream(x).view(-1, self.action_size, self.n_atoms)
 
        # Dueling combination — subtract mean advantage to keep decomposition unique
        Q_logits = V + (A - A.mean(dim=1, keepdim=True))    # (batch, action_size, n_atoms)
 
        # Softmax over the atom dimension → probability distribution per action
        return F.softmax(Q_logits, dim=2)
 
    def q_values(self, state):
        """Scalar Q(s,a) = Σ z_i · p_i(s,a) — used for greedy action selection."""
        probs = self.forward(state)                  # (batch, action_size, n_atoms)
        return (probs * self.atoms).sum(dim=2)        # (batch, action_size)
 
    def reset_noise(self):
        """Re-sample noise in every NoisyLinear layer."""
        for layer in self.shared:
            layer.reset_noise()
        # Walk into the Sequential streams and reset NoisyLinear sub-layers
        for stream in (self.value_stream, self.advantage_stream):
            for layer in stream:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()