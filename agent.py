import numpy as np
import random
import pickle
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#define hyperparameters
BUFFER_SIZE = int(1e5) 
BATCH_SIZE = 128       
GAMMA = 0.99   
TAU = 1e-3            
LR_ACTOR = 1e-4      
LR_CRITIC = 1e-4 
LR_DECAY = 0.999
UPDATE_FREQ = 20
UPDATES = 15
EPSILON = 1.0        
EPSILON_DECAY = 1e-6

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05, NOISE_DECAY = 0.999):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.decay_factor = NOISE_DECAY
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        self.sigma = self.sigma * self.decay_factor
        self.theta = self.theta * self.decay_factor
        return self.state


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Agent():

    def __init__(self, state_size, action_size, random_seed=2020,
                 fc1_units=128, fc2_units=64, epsilon=1.0, lr_actor=1e-3, lr_critic=1e-4, LR_DECAY=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.epsilon = epsilon
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_decay = LR_DECAY

        self.noise = OUNoise(action_size, self.random_seed)
        random.seed(self.random_seed)

        ###define the actor networks, both local and target
        self.actor = Actor(self.state_size, self.action_size, self.random_seed, self.fc1_units, self.fc2_units).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.random_seed, self.fc1_units, self.fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=self.lr_decay)
        
        ### define the critic networks, both local and target
        self.critic = Critic(self.state_size, self.action_size, self.random_seed,
                             self.fc1_units, self.fc2_units).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.random_seed,
                                    self.fc1_units, self.fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=self.lr_decay)
        
        ## create the replaybuffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.random_seed)
        self.t_step = 0
       
    def lr_step(self):
        ## set up for the schedulr, learning rate will decay by steps.
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, tau=1e-3, epsilon_decay=1e-6):
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)

        # Critic update
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # perform gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Actor update
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Targets update
        self.soft_update(self.critic, self.critic_target, tau)
        self.soft_update(self.actor, self.actor_target, tau)

        #  Noise update
        self.epsilon -= epsilon_decay
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_FREQ

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

