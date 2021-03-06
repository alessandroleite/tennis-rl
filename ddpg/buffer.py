from collections import deque, namedtuple

import random
import numpy as np
import torch

class ReplayBuffer(object):
    
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", \
                                                                "other_states", "other_actions", "other_next_states"])
        self.device = device
    
    def add(self, state, action, reward, next_state, done, other_states, other_actions, other_next_states):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, other_states, other_actions, other_next_states)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        return self.convert_to_tensor(experiences)

    def convert_to_tensor(self, experiences):

        """Convert experiences to tensors"""

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        states_others_agents = torch.from_numpy(np.vstack([e.other_states for e in experiences if e is not None])).float().to(self.device)
        actions_other_agents = torch.from_numpy(np.vstack([e.other_actions for e in experiences if e is not None])).float().to(self.device)
        next_states_other_agents = torch.from_numpy(np.vstack([e.other_next_states for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones, states_others_agents, actions_other_agents, next_states_other_agents)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)