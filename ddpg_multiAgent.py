from ddpg_agent import Agent
import torch
import numpy as np
from collections import namedtuple, deque
import random

from constant_params import constParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent:
    def __init__(self, state_size, action_size, random_seed, num_agents):

        self.params = constParams()
        
        self.memory = ReplayBuffer(action_size, self.params.buffer_size, self.params.batch_size, random_seed)

        self.ddpg_agents = [Agent(state_size=state_size, action_size=action_size, memory=self.memory, random_seed=random_seed, num_agents=num_agents) for _ in range(num_agents)]

        self.t_step = 0
        
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
       
    def act(self, state, epsilon, add_noise=True):
        """actions for each agent"""
        action = [agent.act(states, epsilon, add_noise) for agent, states in zip(self.ddpg_agents, state)]
        return action
        
    def step(self, state, action, rewards, next_state, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for stateAg, actionAg, rewardAg, next_stateAg, doneAg in zip(state, action, rewards, next_state, dones):
            self.memory.add(stateAg, actionAg, rewardAg, next_stateAg, doneAg)        

        # Learn, if enough samples are available in memory, num_update times every update_every timesteps
        self.t_step = (self.t_step + 1) % self.params.update_every
        if (self.t_step == 0) and (len(self.memory) > self.params.batch_size):   
            for agent in self.ddpg_agents:
                experiences = self.memory.sample() 
                agent.learn(experiences, self.params.gamma)
        
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
