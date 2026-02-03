import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

import random

from pettingzoo.sisl import pursuit_v4

class QNet(nn.Module):
    def __init__(self, observation_shape, n_actions, hidden_dim=128):
        super(QNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(observation_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = observation_shape[0] * observation_shape[1] * 64
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        # a single observation in a batch costume
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # pettingzoo format -> pytorch format
        # (Batch, Height, Width, Channels) -> (Batch, Channels, Height, Width)
        if len(x.shape) == 4 and x.shape[-1] <= 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        return self.fc(x)


class Buffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, termination):
        self.buffer.append((state, action, reward, next_state, termination))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class MARLAgent:
    def __init__(self, observation_shape, n_actions, learning_rate=1e-3, gamma=0.99, device='cpu'):
        # hyperparams
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        
        # equip networks
        self.q_network = QNet(observation_shape, n_actions).to(device)
        self.target_network = QNet(observation_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # optim
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # buffer
        self.buffer = Buffer()

    # epsilon greedy selection
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    # update network from buffer sample data when enough for a batch
    def update_q(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # TF conversion
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # mean squared loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # optim
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    # update target network from q network weights
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())