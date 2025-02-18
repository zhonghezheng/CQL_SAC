# importing data points
import os
import pickle
import pandas as pd
filename = os.path.join(os.path.dirname(__file__), 'data_augmented.txt')
with open(filename, 'rb') as f:
    data = pickle.load(f)
    data = pd.DataFrame(data)

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import nn 
from torch.optim import Adam
from model import Critic, Actor
from parameters import alpha, rho, gamma, alpha_prime
import numpy as np

class SAC():
    def __init__(self, state_dim=3, action_dim=1, critic_hidden=256, actor_hidden=256):
        self.critic = Critic(action_dim=action_dim, state_dim=state_dim, hidden_dim=critic_hidden)
        self.critic_target = Critic(action_dim=action_dim, state_dim=state_dim, hidden_dim=critic_hidden)

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=actor_hidden)

        # Temperature parameters
        self.target_entropy = -action_dim
        self.alpha = alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.log_alpha_optimizer = Adam(params = [self.log_alpha], lr=0.005)
        
        # Set target parameters equal to main parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.critic_optim = Adam(self.critic.parameters(), lr=0.005)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.005)
        
    def eval(self):
        self.critic.eval() 
        self.critic_target.eval() 
        self.actor.eval() 

    def train(self):
        self.critic.train(True) 
        self.critic_target.train(True) 
        self.actor.train(True)

    def update(self, batchsize):
        # Sample batch of transitions
        # episode,actions,state,next_state,reward,done
        batch = data.sample(batchsize)
        states =  torch.tensor(np.stack(batch['states'].values))
        next_states = torch.tensor(np.stack(batch['next_states'].values))
        actions = torch.tensor(np.stack(batch['actions'].values))
        dones = torch.tensor(np.stack(batch['done'].values))
        rewards = torch.tensor(np.stack(batch['rewards'].values.reshape((-1, 1))))

        # Compute targets for Q functions
        pi_actions_next, pi_log_next = self.actor(next_states)
        q1_next, q2_next = self.critic_target(next_states, pi_actions_next)

        y = rewards + gamma*(1. - dones)*(torch.min(q1_next, q2_next) - self.alpha*pi_log_next)

        # y = rewards + gamma*(1.)*(torch.min(q1_next, q2_next) - self.alpha*pi_log_next)

        q1, q2 = self.critic(states, actions)
    
        # Standard Q-loss
        q1_loss = F.mse_loss(q1, y.detach())
        q2_loss = F.mse_loss(q2, y.detach())

#-----------------------------------------------------------------------
        # CQL stuff
        # actions sampled from uniform distribution
        actions_mu = torch.tensor(np.stack(np.random.uniform(low = -2, high = 2, size = (batchsize,1)).astype(np.float32)))
        pi_actions, pi_log = self.actor(states)
        pi_actions_next, pi_log_next = self.actor(next_states)
        q1_mu, q2_mu = self.critic(states, actions_mu)
        q1_pi, q2_pi = self.critic(states, pi_actions)
        q1_next, q2_next = self.critic(next_states, pi_actions_next)

        # importance sampled version, helps with higher dimension 
        random_density = np.log(0.5 ** pi_actions.shape[-1])
        cat_q1 = torch.cat(
            [q1_mu - random_density, q1_next - pi_log_next, q1_pi - pi_log], 1
        )
        cat_q2 = torch.cat(
            [q2_mu - random_density, q2_next - pi_log_next, q2_pi - pi_log], 1
        )

        # Using variant of CQL with KL-divergence for regularization
        min_qf1_loss = torch.logsumexp(cat_q1, dim=1,).mean() 
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1,).mean() 

        min_qf1_loss = min_qf1_loss - q1.mean()
        min_qf2_loss = min_qf2_loss - q2.mean()

        # min_qf1_loss = torch.logsumexp(q1_mu, dim=1,).mean() 
        # min_qf2_loss = torch.logsumexp(q2_mu, dim=1,).mean() 

        # min_qf1_loss = min_qf1_loss - q1_pi.mean()
        # min_qf2_loss = min_qf2_loss - q2_pi.mean()

        # Update Q-functions via gradient descent
        q_loss = 0.5*q1_loss + 0.5*q2_loss + alpha_prime * min_qf1_loss + alpha_prime *min_qf2_loss

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

#-----------------------------------------------------------------------

        # Update policy via gradient descent
        phi_actions, phi_log = self.actor(states)

        q1_predict, q2_predict = self.critic(states, phi_actions)
        phi_loss = -(torch.min(q1_predict, q2_predict) - self.alpha*phi_log).mean()

        self.actor_optim.zero_grad()
        phi_loss.backward()
        self.actor_optim.step()

        # Update temperature parameter 
        alpha_loss = -((self.log_alpha.exp() * (phi_log + self.target_entropy).detach().mean()))
        self.log_alpha_optimizer.zero_grad() 
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

#-----------------------------------------------------------------------
        # Update target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - rho) + param.data * rho)

    
    def get_mean(self, state):
        phi_actions = self.actor.get_mean(state)
        return phi_actions

