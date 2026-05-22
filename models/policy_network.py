import torch
import torch.nn as nn

class Policy_Network(nn.Module):
    """Parametrized Policy Network.
     A policy is a mapping from the current environment observation to a probability distribution of the actions to be taken.
     The policy used in the tutorial is parameterized by a neural network. It consists of 2 linear layers that are shared
     between both the predicted mean and standard deviation.
        Further, the single individual linear layers are used to estimate the mean and the standard deviation.
        nn.Tanh is used as a non-linearity between the hidden layers. The following function estimates a mean
        and standard deviation of a normal distribution from which an action is sampled. Hence it is expected
        for the policy to learn appropriate weights to output means and standard deviation based on the
        current observation.
    """

    def __init__(self, state_size: int, action_size: int, layers=[16, 32], policy_type="discrete"):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        self.policy_type = policy_type

        layer_sizes = [state_size] + layers
        self.shared_net = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        if self.policy_type == "discrete":
            # outputs action logits for Categorical distribution
            self.policy_head = nn.Linear(layer_sizes[-1], action_size)
        elif self.policy_type == "continuous":
            # outputs mean and learnable log_std for Normal distribution
            # Policy Mean specific Linear Layer
            self.mean_head = nn.Linear(layer_sizes[-1], action_size)
            # Policy Std Dev specific Linear Layer
            self.log_std = nn.Parameter(torch.zeros(action_size))


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        # shared_features = self.shared_net(x.float())

        x = x.float()

        for layer in self.shared_net:
            x = torch.tanh(layer(x))

        if self.policy_type == "discrete":
            logits = self.policy_head(x)
            return logits
        else:
            action_means = self.mean_head(x)
            action_stds = torch.exp(self.log_std).expand_as(action_means)
            return action_means, action_stds

