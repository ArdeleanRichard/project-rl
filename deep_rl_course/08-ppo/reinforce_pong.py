import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import pong_utils
from utils.parallelEnv import parallelEnv

# widget bar to display progress
import progressbar as pb


# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        ########
        ##
        ## Modify your neural network
        ##
        ########

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # output = 20x20 here
        self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
        self.size = 1 * 20 * 20

        # 1 fully connected layer
        self.fc = nn.Linear(self.size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ########
        ##
        ## Modify your neural network
        ##
        ########

        x = F.relu(self.conv(x))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))


def surrogate(policy, old_probs, states, actions, rewards, discount=0.995, beta=0.01):
    ########
    ##
    ## WRITE YOUR OWN CODE HERE
    ##
    ########

    actions = torch.tensor(actions, dtype=torch.int8, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    return torch.mean(beta * entropy)


def train(policy, optimizer):
    # training loop max iterations
    episode = 500
    # episode = 800             # WARNING: running through all 800 episodes will take 30-45 minutes

    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    # initialize environment
    # envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)
    envs = parallelEnv("ALE/Pong-v5", n=8, seed=1234)

    discount_rate = .99
    beta = .01
    tmax = 320

    # keep track of progress
    mean_rewards = []

    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        # L = -surrogate(policy, old_probs, states, actions, rewards, discount=discount_rate, beta=beta)

        L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, discount=discount_rate, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print(f"Episode: {e + 1:d}, score: {np.mean(total_rewards):f}")
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    timer.finish()

    return mean_rewards