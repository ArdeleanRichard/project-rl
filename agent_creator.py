from agent.agent_dqn import AgentDQN
from agent.agent_q_learning import AgentQLearning
from agent.agent_sarsa_expected import AgentSarsaExpected
from agent.agent_sarsa_zero import AgentSarsaZero


class AgentCreator:
    def __init__(self, env, config=None):
        self.env = env
        self.config = config

    def create(self):
        # if self.config["name"] == "q_learning":
        if "q_learning" in self.config["name"]:
            return AgentQLearning(env=self.env, config=self.config)

        # if self.config["name"] == "sarsa_zero":
        if "sarsa_zero"in self.config["name"]:
            return AgentSarsaZero(env=self.env, config=self.config)

        # if self.config["name"] == "sarsa_expected":
        if "sarsa_expected" in self.config["name"]:
            return AgentSarsaExpected(env=self.env, config=self.config)

        if "dqn" in self.config["name"]:
            return AgentDQN(env=self.env, config=self.config)