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
        # Define the three layers here
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, a_size)

    def forward(self, x):
        # Define the forward process here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)





def run_pixelcopter():
    env_id = "Pixelcopter-PLE-v0"
    env, s_size, a_size = create_env(env_id)

    pixelcopter_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 50000,
        "n_evaluation_episodes": 10,
        "max_t": 10000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    # Create policy and place it to the device
    cartpole_policy = Policy(pixelcopter_hyperparameters["state_space"], pixelcopter_hyperparameters["action_space"], pixelcopter_hyperparameters["h_size"]).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

    scores = reinforce(
        env,
        cartpole_policy,
        cartpole_optimizer,
        pixelcopter_hyperparameters["n_training_episodes"],
        pixelcopter_hyperparameters["max_t"],
        pixelcopter_hyperparameters["gamma"],
        1000
    )

    evaluate_agent(
        env_id,
        cartpole_policy,
        pixelcopter_hyperparameters["n_evaluation_episodes"],
        pixelcopter_hyperparameters["max_t"]
    )


if __name__ == "__main__":
    run_pixelcopter()