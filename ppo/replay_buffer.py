import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", \
            field_names=["state", "action_took", "old_actions_prob", "reward"])
        self.seed = random.seed(seed)
    
    def add(self, state, action_took, old_actions_prob, reward):
        """Add a new experience to memory."""
        e = self.experience(state, action_took, old_actions_prob, reward)
        self.memory.append(e)

    def adds(self, states, action_tooks, old_actions_probs, rewards):
        """Add a new experience to memory."""
        for i in range(rewards.shape[0]):
            e = self.add(states[i], action_tooks[i], old_actions_probs[i], rewards[i])
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.stack([e.state for e in experiences if e is not None])
        action_tooks = np.stack([e.action_took for e in experiences if e is not None])
        old_actions_probs = np.stack([e.old_actions_prob for e in experiences if e is not None])
        rewards = np.stack([e.reward for e in experiences if e is not None]).reshape((self.batch_size, 1))
  
        return states, action_tooks, rewards, old_actions_probs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory) 