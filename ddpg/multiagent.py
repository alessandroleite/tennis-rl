from typing import Mapping

import numpy as np
import random

import torch

from ddpg.agent import DDPGAgent
from ddpg.buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent(object):

    def __init__(self, num_agents, state_size, action_size, seed, hyperparameters:Mapping[str, float]):
        """Initialize a multi-agent object """
        self.action_size = action_size 
        self.__name__ = 'MADDPG'
        memory = ReplayBuffer(action_size, hyperparameters['buffer_size'], device)
        self.agents = [DDPGAgent(id, state_size, action_size, seed, memory, num_agents, hyperparameters) for id in range(num_agents)] 
        

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.agents, states, actions, rewards, next_states, dones)
        agents_ids = np.arange(len(self.agents))

        for id, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            other_states = states[agents_ids[agents_ids != id]] # agents_ids != agent.id
            other_actions = actions[agents_ids[agents_ids != id]]
            other_next_states = next_states[agents_ids[agents_ids != id]]
            agent.step(state, action, reward, next_state, done, other_states, other_actions, other_next_states)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy.""" 

        actions = np.zeros(shape=(len(self.agents), self.action_size))
        for agent in self.agents:
            actions[agent.id, :] = agent.act(states[agent.id], add_noise)
        return actions
    
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()
    
    def __len__(self):
        return len(self.agents)

    def __getitem__(self, id):
        return self.agents[id]
