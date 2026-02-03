import torch
import torch.nn as nn
import torch.optim as optim

import random

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


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs_buf = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.rewards_buf = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def push_tensor(self, obs, action, reward, next_obs, done):
        device = self.obs_buf.device
        obs_tensor = torch.tensor(obs, dtype=self.obs_buf.dtype, device=device)
        next_obs_tensor = torch.tensor(next_obs, dtype=self.next_obs_buf.dtype, device=device)
        action_tensor = torch.tensor(action, dtype=self.actions_buf.dtype, device=device)
        reward_tensor = torch.tensor(reward, dtype=self.rewards_buf.dtype, device=device)
        done_tensor = torch.tensor(done, dtype=self.done_buf.dtype, device=device)

        idx = self.pos
        self.obs_buf[idx] = obs_tensor
        self.next_obs_buf[idx] = next_obs_tensor
        self.actions_buf[idx] = action_tensor
        self.rewards_buf[idx] = reward_tensor
        self.done_buf[idx] = done_tensor

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = dict(
            obs=self.obs_buf[idxs],
            actions=self.actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs]
        )
        return batch

    def to_device(self, device):
        self.device = device
        self.obs_buf = self.obs_buf.to(device)
        self.next_obs_buf = self.next_obs_buf.to(device)
        self.actions_buf = self.actions_buf.to(device)
        self.rewards_buf = self.rewards_buf.to(device)
        self.done_buf = self.done_buf.to(device)

    def __len__(self):
        return self.size

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
        self.buffer = ReplayBuffer(capacity=10000, obs_shape=observation_shape, device=device)

    # epsilon greedy selection
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.from_numpy(state).float().to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    # update network from buffer sample data when enough for a batch
    def update_q(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0
        
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return 0.0
    
        states = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_obs']
        dones = batch['done']

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