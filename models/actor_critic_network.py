import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, layers=[256]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        layer_sizes = [state_size] + layers + [action_size]
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fcs[:-1]:
            fc.weight.data.uniform_(*hidden_init(fc))
        self.fcs[:-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return F.tanh(self.fcs[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, layers=[256, 256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        layer_sizes = [state_size] + layers + [1]
        layer_sizes[1] += + action_size
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        self.reset_parameters()

    def forward(self, state, action):
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        for fc in self.fcs[1:-1]:
            x = F.leaky_relu(fc(x))
        return self.fcs[-1](x)

    def reset_parameters(self):
        for fc in self.fcs[:-1]:
            fc.weight.data.uniform_(*hidden_init(fc))
        self.fcs[:-1].weight.data.uniform_(-3e-3, 3e-3)





class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for A2C.

    The actor outputs a probability distribution over actions (policy).
    The critic outputs a value estimate V(s) of the current state.
    Shares early layers for feature extraction, then splits into two heads.
    """

    def __init__(self, state_size, action_size, layers=[64, 64]):
        super(ActorCriticNetwork, self).__init__()

        self.action_size = action_size

        # Shared feature extractor
        shared_sizes = [state_size] + layers
        self.shared_layers = nn.ModuleList([
            nn.Linear(shared_sizes[i], shared_sizes[i + 1])
            for i in range(len(shared_sizes) - 1)
        ])

        hidden = layers[-1]

        # Actor head: outputs action probabilities
        self.actor = nn.Linear(hidden, action_size)

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        """
        Returns both policy logits and state value.

        Returns:
            action_logits: (batch, action_size) - unnormalized log probabilities
            state_value: (batch, 1) - V(s) estimate
        """
        x = state
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value

    def get_action_probs(self, state):
        """Returns softmax probabilities over actions."""
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)

    def get_value(self, state):
        """Returns only the state value estimate."""
        _, state_value = self.forward(state)
        return state_value


class SeparateActorCriticNetwork(nn.Module):
    """
    Separate Actor-Critic Networks for A2C.

    Unlike the shared version, this uses completely separate networks
    for the actor and critic. Can be useful when the optimal features
    for policy and value differ significantly.
    """

    def __init__(self, state_size, action_size, actor_layers=[64, 64], critic_layers=[64, 64]):
        super(SeparateActorCriticNetwork, self).__init__()

        self.action_size = action_size

        # Actor network
        actor_sizes = [state_size] + actor_layers + [action_size]
        self.actor_layers = nn.ModuleList([
            nn.Linear(actor_sizes[i], actor_sizes[i + 1])
            for i in range(len(actor_sizes) - 1)
        ])

        # Critic network
        critic_sizes = [state_size] + critic_layers + [1]
        self.critic_layers = nn.ModuleList([
            nn.Linear(critic_sizes[i], critic_sizes[i + 1])
            for i in range(len(critic_sizes) - 1)
        ])

    def forward(self, state):
        """Returns both policy logits and state value."""
        # Actor forward pass
        x_actor = state
        for layer in self.actor_layers[:-1]:
            x_actor = F.relu(layer(x_actor))
        action_logits = self.actor_layers[-1](x_actor)

        # Critic forward pass
        x_critic = state
        for layer in self.critic_layers[:-1]:
            x_critic = F.relu(layer(x_critic))
        state_value = self.critic_layers[-1](x_critic)

        return action_logits, state_value

    def get_action_probs(self, state):
        """Returns softmax probabilities over actions."""
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)

    def get_value(self, state):
        """Returns only the state value estimate."""
        _, state_value = self.forward(state)
        return state_value