import torch
from torch import nn
from math import exp 
from torch.distributions import Normal

# Learns 2 Q function: given action and state, what is the expected cumulative reward
class Critic(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.Q1 = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = self.flatten(torch.concatenate([state, action], 1))
        val1 = self.Q1(x)
        val2 = self.Q2(x)
        return val1, val2
    
# given state, what should be the action taken
# in sac, policy learns both the mean and std for exploration
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)

        # using log std for numerical stability (less sensitive to numerical roundoffs)
        # assuming Normal distribution?
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.flatten(state)
        x = self.policy(x)
        mean = self.mean_linear(x)
        # mean = torch.tanh(mean) * 2
        log_std = self.log_std_linear(x)
        return mean, log_std
    
    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = exp(log_std)
        
        # TODO: change this to be with varying a 
        normal_distribution = Normal(mean, std)

        #rsample allows us to back-propogate
        action = normal_distribution.rsample()

        return action, log_std
    
    def get_mean(self, state):
        mean, log = self.forward(state)
        return mean

        

        