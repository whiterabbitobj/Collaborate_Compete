# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    """
    Standard replay buffer to hold memories for later learning. Returns
    random experiences with no consideration of their usefulness.

    When using an agent with a ROLLOUT trajectory, then instead of each
    experience holding SARS' data, it holds:
    state = state at t
    action = action at t
    reward = cumulative reward from t through t+n-1
    next_state = state at t+n

    where n=ROLLOUT.
    """
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5,
                 agent_count=1):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout
        self.agent_count = agent_count

    def store(self, experience):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """

        if self.rollout > 1:
            self.n_step.append(experience)
            # Abort if ROLLOUT steps haven't been taken in a new episode
            if len(self.n_step) < self.rollout:
                return
            experience = self._n_stack()

        obs, next_obs, actions, rewards, dones = experience
        actions = torch.from_numpy(np.concatenate(actions)).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.tensor(dones).float()
        self.buffer.append((obs, next_obs, actions, rewards, dones))

    def sample(self, batch_size):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """

        batch = random.sample(self.buffer, k=batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)

        # Transpose the num_agents and batch_size, for easy indexing later
        # e.g. from 64 samples of 2 agents each, to 2 agents with 64 samples
        obs = torch.stack(obs).transpose(1,0).to(self.device)
        next_obs = torch.stack(next_obs).transpose(1,0).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).transpose(1,0).to(self.device)
        dones = torch.stack(dones).transpose(1,0).to(self.device)
        return (obs, next_obs, actions, rewards, dones)

    def init_n_step(self):
        """
        Creates (or rezeroes an existing) deque to handle nstep returns.
        """

        self.n_step = deque(maxlen=self.rollout)

    def _n_stack(self):
        """
        Takes a stack of experience tuples of depth ROLLOUT, and calculates
        the discounted real rewards, then returns the next_obs at ROLLOUT
        timesteps to be used with a nstep trajectory structure Q value.
        """

        obs, next_obs, actions, rewards, dones = zip(*self.n_step)

        summed_rewards = rewards[0]
        for i in range(1, self.rollout):
            summed_rewards += self.gamma**i * rewards[i]
            if np.any(dones[i]):
                break

        obs = obs[0]
        nstep_obs = next_obs[i]
        actions = actions[0]
        dones = dones[i]
        return (obs, nstep_obs, actions, summed_rewards, dones)

    def __len__(self):
        return len(self.buffer)
