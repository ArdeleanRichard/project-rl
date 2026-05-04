import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla Q-Network
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers=[64, 64]):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        layer_sizes = [state_size] + layers + [action_size]
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, state):
        x = state
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x)


# ──────────────────────────────────────────────────────────────────────────────
# Dueling Q-Network
# ──────────────────────────────────────────────────────────────────────────────

class QNetworkDueling(nn.Module):
    """
    Dueling architecture (Wang et al., 2016):
      Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    Subtracting the mean advantage makes V and A uniquely identifiable.
    """

    def __init__(self, state_size, action_size, seed, layers=[64, 64]):
        super(QNetworkDueling, self).__init__()
        self.seed = torch.manual_seed(seed)

        shared = [state_size] + layers
        self.shared = nn.ModuleList([
            nn.Linear(shared[i], shared[i + 1])
            for i in range(len(shared) - 1)
        ])

        hidden = layers[-1]
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_size)
        )

    def forward(self, state):
        x = state
        for fc in self.shared:
            x = F.relu(fc(x))
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        return V + (A - A.mean(dim=1, keepdim=True))


# ──────────────────────────────────────────────────────────────────────────────
# Distributional Q-Network (C51)
# ──────────────────────────────────────────────────────────────────────────────

class QNetworkDistributional(nn.Module):
    """
    C51 - Distributional RL (Bellemare et al., 2017).

    Predicts a probability distribution over N discrete return "atoms"
    z_i evenly spaced in [v_min, v_max], instead of a scalar Q-value.

      p(s,a) = softmax( network(s) )   shape: (batch, action_size, n_atoms)
      Q(s,a) = sum_i z_i * p_i(s,a)   (expected return, used for action selection)

    CRITICAL - v_min / v_max must bracket the full *discounted return* range:
    For LunarLander (gamma=0.99, episodes up to 1000 steps), returns span
    roughly [-300, +300]. Using [-10, 10] clips 100% of atoms for bad episodes,
    making the gradient signal completely uninformative.
    """

    def __init__(self, state_size, action_size, seed,
                 layers=[64, 64], n_atoms=51, v_min=-200.0, v_max=200.0):
        super().__init__()
        self.seed        = torch.manual_seed(seed)
        self.action_size = action_size
        self.n_atoms     = n_atoms

        # Fixed support — register as buffer so it moves with .to(device)
        # without being a trainable parameter.
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))

        layer_sizes = [state_size] + layers
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        # Output one logit per (action, atom) pair
        self.output = nn.Linear(layers[-1], action_size * n_atoms)

    def forward(self, state):
        """
        Returns log-softmax over atoms for every action.
        Shape: (batch, action_size, n_atoms)
        Returning log-probs (not raw probs) lets the cross-entropy loss
        -(m * log_p).sum() remain numerically stable.
        """
        x = state
        for fc in self.fcs:
            x = F.relu(fc(x))
        logits = self.output(x).view(-1, self.action_size, self.n_atoms)
        return F.log_softmax(logits, dim=2)

    def get_probs(self, state):
        """Softmax probabilities. Shape: (batch, action_size, n_atoms)."""
        return self.forward(state).exp()

    def q_values(self, state):
        """Expected Q-values: Q(s,a) = sum_i z_i * p_i. Shape: (batch, action_size)."""
        return (self.get_probs(state) * self.atoms).sum(dim=2)


# ──────────────────────────────────────────────────────────────────────────────
# NoisyLinear
# ──────────────────────────────────────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear layer (Fortunato et al., 2017).

    Each weight and bias has a learnable mean (mu) and std (sigma) plus
    factorised Gaussian noise (epsilon) resampled each step.

      w = mu_w + sigma_w * eps_w,   b = mu_b + sigma_b * eps_b

    Factorised noise: draw p+q scalars, combine via outer product -> O(p+q).
    Sigma is learned: the network decides per-state how much noise helps,
    and anneals sigma toward zero as it gains confidence. This replaces
    the epsilon-greedy schedule entirely.

    Call reset_noise() once per learning step (and once per act step for
    Noisy DQN). Do NOT call it inside forward() — you control the timing.
    At eval() only mu is used (deterministic greedy).
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self._sigma_init  = sigma_init

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self._sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self._sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        """f(x) = sgn(x)*sqrt(|x|) — factorised noise transform from the paper."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ──────────────────────────────────────────────────────────────────────────────
# Noisy Q-Network
# ──────────────────────────────────────────────────────────────────────────────

class QNetworkNoisy(nn.Module):
    """
    Noisy DQN (Fortunato et al., 2017).

    The first hidden layer(s) use standard Linear — adding noise to the deep
    shared feature extractor introduces instability without exploration benefit.
    Only the last hidden layer and output layer are NoisyLinear.

    For layers=[64,64]:
      Linear(state->64) -> ReLU -> NoisyLinear(64->64) -> ReLU -> NoisyLinear(64->actions)

    No epsilon-greedy is needed. reset_noise() must be called once per act()
    call AND once per learn() call to resample noise each step.
    """

    def __init__(self, state_size, action_size, seed,
                 layers=[64, 64], sigma_init=0.5):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Plain layers: all hidden sizes except the last one
        plain_sizes = [state_size] + layers[:-1]
        self.feature_layers = nn.ModuleList([
            nn.Linear(plain_sizes[i], plain_sizes[i + 1])
            for i in range(len(plain_sizes) - 1)
        ])

        # NoisyLinear head
        in_noisy = layers[-2] if len(layers) >= 2 else state_size
        hidden   = layers[-1]
        self.noisy1 = NoisyLinear(in_noisy, hidden,      sigma_init)
        self.noisy2 = NoisyLinear(hidden,   action_size, sigma_init)

    def forward(self, state):
        x = state
        for fc in self.feature_layers:
            x = F.relu(fc(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


# ──────────────────────────────────────────────────────────────────────────────
# Rainbow Q-Network  (Dueling + Distributional + Noisy)
# ──────────────────────────────────────────────────────────────────────────────

class QNetworkRainbow(nn.Module):
    """
    Rainbow network (Hessel et al., 2017).

    Three architectural improvements fused into one network:
      - Dueling streams: separate V(s) and A(s,a) heads
      - Distributional: each head outputs n_atoms logits (not a scalar)
      - NoisyLinear: all head layers use learnable noise (no epsilon-greedy)

    The shared extractor uses plain Linear for stability.
    Output shape: (batch, action_size, n_atoms) log-softmax probabilities.
    """

    def __init__(self, state_size, action_size, seed,
                 layers=[64, 64], n_atoms=51, v_min=-200.0, v_max=200.0,
                 sigma_init=0.5):
        super().__init__()
        self.seed        = torch.manual_seed(seed)
        self.action_size = action_size
        self.n_atoms     = n_atoms

        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))

        shared_sizes = [state_size] + layers
        self.shared = nn.ModuleList([
            nn.Linear(shared_sizes[i], shared_sizes[i + 1])
            for i in range(len(shared_sizes) - 1)
        ])

        hidden = layers[-1]

        # Value stream: hidden -> n_atoms (one distribution over returns)
        self.value_noisy1 = NoisyLinear(hidden, hidden,  sigma_init)
        self.value_noisy2 = NoisyLinear(hidden, n_atoms, sigma_init)

        # Advantage stream: hidden -> action_size * n_atoms
        self.adv_noisy1 = NoisyLinear(hidden, hidden,                sigma_init)
        self.adv_noisy2 = NoisyLinear(hidden, action_size * n_atoms, sigma_init)

    def forward(self, state):
        x = state
        for fc in self.shared:
            x = F.relu(fc(x))

        v = F.relu(self.value_noisy1(x))
        v = self.value_noisy2(v).view(-1, 1, self.n_atoms)

        a = F.relu(self.adv_noisy1(x))
        a = self.adv_noisy2(a).view(-1, self.action_size, self.n_atoms)

        # Dueling combination per-atom, then log-softmax over atoms
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))
        return F.log_softmax(q_atoms, dim=2)

    def get_probs(self, state):
        return self.forward(state).exp()

    def q_values(self, state):
        return (self.get_probs(state) * self.atoms).sum(dim=2)

    def reset_noise(self):
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()