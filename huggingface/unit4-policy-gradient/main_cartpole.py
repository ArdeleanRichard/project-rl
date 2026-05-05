# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from reinforce import create_env, reinforce, evaluate_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def run_cartpole():
    env_id = "CartPole-v1"
    env, s_size, a_size = create_env(env_id)

    cartpole_hyperparameters = {
        "h_size": 16,
        "n_training_episodes": 1000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    # Create policy and place it to the device
    cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    scores = reinforce(
        env,
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        100
    )

    evaluate_agent(
        env_id,
        cartpole_policy,
        cartpole_hyperparameters["n_evaluation_episodes"],
        cartpole_hyperparameters["max_t"]
    )


if __name__ == "__main__":
    run_cartpole()