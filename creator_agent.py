from agent.agent_a2c import AgentA2C
from agent.agent_ddpg import AgentDDPG
from agent.agent_dqn import AgentDQN
from agent.agent_dqn_distributional import AgentDistributionalDQN
from agent.agent_dqn_double import AgentDoubleDQN
from agent.agent_dqn_dueling import AgentDuelingDQN
from agent.agent_dqn_prioritized import AgentPriorityDQN
from agent.agent_dqn_rainbow import AgentRainbow
from agent.agent_q_learning import AgentQLearning
from agent.agent_reinforce import AgentReinforce
from agent.agent_sarsa_expected import AgentSarsaExpected
from agent.agent_sarsa_zero import AgentSarsaZero
from agent.agent_dqn_noisy import AgentNoisyDQN


class AgentCreator:
    def __init__(self, env, config=None):
        self.env = env
        self.config = config

        if "policy_type" in self.env.config:
            self.config["policy_type"] = self.env.config["policy_type"]

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

        if "dqn_double" in self.config["name"]:
            return AgentDoubleDQN(env=self.env, config=self.config)

        if "dqn_dueling" in self.config["name"]:
            return AgentDuelingDQN(env=self.env, config=self.config)

        if "dqn_priority" in self.config["name"]:
            return AgentPriorityDQN(env=self.env, config=self.config)

        if "dqn_distributional" in self.config["name"]:
            return AgentDistributionalDQN(env=self.env, config=self.config)

        if "dqn_noisy" in self.config["name"]:
            return AgentNoisyDQN(env=self.env, config=self.config)

        if "dqn_rainbow" in self.config["name"]:
            return AgentRainbow(env=self.env, config=self.config)

        if "dqn" in self.config["name"]:
            return AgentDQN(env=self.env, config=self.config)

        if "reinforce" in self.config["name"]:
            return AgentReinforce(env=self.env, config=self.config)

        if "ddpg" in self.config["name"]:
            return AgentDDPG(env=self.env, config=self.config)

        if "a2c" in self.config["name"]:
            return AgentA2C(env=self.env, config=self.config)