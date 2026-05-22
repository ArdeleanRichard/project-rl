import torch.optim as optim

from agent.agent_dqn import AgentDQN
from models.q_network import QNetworkDueling


class AgentDuelingDQN(AgentDQN):
    """
    Dueling Network DQN (Wang et al., 2016).

    Separate V(s) and A(s,a) streams. Q = V + (A - mean(A)).
    learn() is identical to vanilla DQN — only the architecture changes.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.qnetwork_local  = QNetworkDueling(self.n_states, self.n_actions).to(self.device)
        self.qnetwork_target = QNetworkDueling(self.n_states, self.n_actions).to(self.device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)