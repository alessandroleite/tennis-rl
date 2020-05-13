import random
import numpy as np

from typing import Mapping

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg.models import Critic, Actor
from ddpg.noise import Ornstein

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch_size=200, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, weight_decay=0, tau=1e-3, update_frequency=20, n_learns=1

class DDPGAgent(object):
    """Interacts with and learns from the environment."""
    
    def __init__(self, id, state_size, action_size, seed, memory, num_agents, hyperparameters:Mapping[str, float]):
        """Initialize a DDPG agent object.
        
        Params
        ======
            id (int): agent's id
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            memory (ReplayBuffer): replay buffer to store the experience of this agent
            hyperparameters (dictionnary of str:): hyperparameters' values of the model. The expected parameters are:
             - batch_size (int): minibatch size
             - lr_actor (float): learning rate of the actor 
             - lr_critic (float): learning rate of the critic 
             - gamma (float): discount factor
             - weight_decay (float): critic L2 weight decay 
             - tau (float): value for soft update of target parameters
             - update_frequency (int): how much steps must be executed before starting learn
             - n_learns (int): how many learning for update 
        """
        self.id = id
        self.__name__ = 'DDPG'
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = hyperparameters['gamma']
        self.batch_size = int(hyperparameters['batch_size'])
        self.tau = hyperparameters['tau']
        
        self.update_frequency = int(hyperparameters['update_frequency'])
        self.n_learns = int(hyperparameters['n_learns'])

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyperparameters['lr_actor'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyperparameters['lr_critic'], weight_decay=hyperparameters['weight_decay'])

        # Noise process
        self.noise = Ornstein(action_size)

        # Replay memory
        self.memory = memory

        # Initialize the time step (for every update_frequency steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, other_states, other_actions, other_next_states):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done, other_states, other_actions, other_next_states)

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            for _ in range(self.n_learns):
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, _, _, _, _, other_states, _, _ = experiences

        self.update_critic(experiences, gamma)
        self.update_actor(states, other_states)
        self.update_target_networks()            

    def update_critic(self, experiences, gamma):

        """Update the critic network given the experiences"""

        states, actions, rewards, next_states, dones, other_states, other_actions, other_next_states = experiences

        all_states = torch.cat((states, other_states), dim=1).to(device)
        all_actions = torch.cat((actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((next_states, other_next_states), dim=1).to(device)

        local_all_next_actions = []
        local_all_next_actions.append(self.actor_target(states))
        local_all_next_actions.append(self.actor_target(other_states))
        all_next_actions = torch.cat(local_all_next_actions, dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def update_actor(self, states, other_states):

        all_states = torch.cat((states, other_states), dim=1).to(device)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        other_actions_pred = self.actor_local(other_states)
        other_actions_pred = other_actions_pred.detach()

        actions_pred = torch.cat((actions_pred, other_actions_pred), dim=1).to(device)
        actor_loss   = -self.critic_local(all_states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def update_target_networks(self):

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)  
    
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)