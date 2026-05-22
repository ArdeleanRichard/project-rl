import numpy as np
import torch

from agent.agent_dqn import AgentDQN, ReplayBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Prioritized DQN
# ──────────────────────────────────────────────────────────────────────────────

class AgentPriorityDQN(AgentDQN):
    """
    Prioritized Experience Replay DQN (Schaul et al., 2015).

    Samples high-TD-error transitions more often. IS weights correct the bias.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.memory = PrioritizedReplayBuffer(self.config)

    def update(self, state, info, action, reward, done, next_state):
        self.memory.add(state, action, reward, done, next_state)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            experiences, indices, weights = self.memory.sample()
            self.learn(experiences, indices, weights)

    def learn(self, experiences, indices, weights):
        states, actions, rewards, dones, next_states = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets      = rewards + self.GAMMA * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        td_errors = (Q_expected - Q_targets).detach().squeeze().cpu().numpy()
        loss      = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none').squeeze()).mean()
        self.memory.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (Schaul et al., 2015).

    Samples transitions proportional to their TD error:
      P(i) = p_i^alpha / sum_k p_k^alpha
    Importance-sampling weights correct the resulting bias:
      w_i = (N * P(i))^{-beta}
    New transitions receive max priority so they are sampled at least once.
    """

    def __init__(self, config, alpha=0.6):
        super().__init__(config)
        self.alpha       = alpha
        self.pos         = 0
        self.buffer_size = self.config["buffer_size"]
        self.priorities  = np.zeros((self.config["buffer_size"],), dtype=np.float32)
        self.memory      = []   # list for O(1) index access

    def add(self, state, action, reward, done, next_state):
        max_prio = self.priorities.max() if self.memory else 1.0
        e = self.experience(state, action, reward, done, next_state)
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
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences])).float().to(self.device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences])).long().to(self.device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences]).astype(np.uint8)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        return (states, actions, rewards, dones, next_states), indices, weights

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(float(err)) + 1e-6

