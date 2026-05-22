from agent.agent_dqn import AgentDQN
import torch.nn.functional as F

class AgentDoubleDQN(AgentDQN):
    """
    Double DQN (van Hasselt et al., 2015).

    Standard DQN uses the target net to both select AND evaluate the best next
    action, causing systematic overestimation. Fix: local net selects the action,
    target net evaluates it.
    """

    def __init__(self, env, config):
        super().__init__(env, config)

    def learn(self, experiences):
        states, actions, rewards, dones, next_states = experiences

        best_actions   = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        Q_targets      = rewards + self.GAMMA * Q_targets_next * (1 - dones)
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.training_error.append(loss.detach().cpu().item())