import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class AgentQNetwork(nn.Module):    
    def __init__(self, observation_shape, n_actions, hidden_dim=64):
        super(AgentQNetwork, self).__init__()
        
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
        # a single observation in a batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # pettingzoo format -> pytorch format
        # (Batch, Height, Width, Channels) -> (Batch, Channels, Height, Width)
        if len(x.shape) == 4 and x.shape[-1] <= 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        return self.fc(x)


class QMixNetwork(nn.Module):    
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(QMixNetwork, self).__init__()
        self.n_agents = n_agents
        
        # set up hypernetworks for mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # set up hypernetworks for mixing network biases
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, agent_qs, state):
        batch_size = agent_qs.shape[0]
        
        # Generate mixing network (positive weights) and (biases)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, -1)
        
        b1 = self.hyper_b1(state).view(batch_size, 1, -1)
        
        # layer 01
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        # layer 02
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, -1, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        # total_q_value
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, -1)


class QMIXReplayBuffer:    
    def __init__(self, capacity, n_agents, obs_shape, device='cpu'):
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.pos = 0
        self.size = 0
        
        # observations for each agent
        self.obs_buf = torch.zeros((capacity, n_agents, *obs_shape), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((capacity, n_agents, *obs_shape), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((capacity, n_agents), dtype=torch.int64, device=device)
        self.rewards_buf = torch.zeros((capacity, n_agents), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((capacity, n_agents), dtype=torch.float32, device=device)
        
        # flatten => all observations
        global_state_dim = n_agents * np.prod(obs_shape)
        self.global_state_buf = torch.zeros((capacity, global_state_dim), dtype=torch.float32, device=device)
        self.next_global_state_buf = torch.zeros((capacity, global_state_dim), dtype=torch.float32, device=device)

    def push_tensor(self, obs_dict, actions_dict, rewards_dict, next_obs_dict, dones_dict):
        device = self.obs_buf.device        
        agents = sorted(obs_dict.keys())
        
        obs_list = [obs_dict[agent] for agent in agents]
        next_obs_list = [next_obs_dict[agent] for agent in agents]
        actions_list = [actions_dict[agent] for agent in agents]
        rewards_list = [rewards_dict[agent] for agent in agents]
        dones_list = [dones_dict[agent] for agent in agents]
        
        # convert to tensors
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions_list, dtype=torch.int64, device=device)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones_list, dtype=torch.float32, device=device)
        
        # flatten and concatenate all observations
        global_state = torch.tensor(np.concatenate([obs.flatten() for obs in obs_list]), 
                                    dtype=torch.float32, device=device)
        next_global_state = torch.tensor(np.concatenate([obs.flatten() for obs in next_obs_list]),
                                        dtype=torch.float32, device=device)
        
        idx = self.pos
        self.obs_buf[idx] = obs_tensor
        self.next_obs_buf[idx] = next_obs_tensor
        self.actions_buf[idx] = actions_tensor
        self.rewards_buf[idx] = rewards_tensor
        self.done_buf[idx] = dones_tensor
        self.global_state_buf[idx] = global_state
        self.next_global_state_buf[idx] = next_global_state
        
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
            done=self.done_buf[idxs],
            global_state=self.global_state_buf[idxs],
            next_global_state=self.next_global_state_buf[idxs]
        )
        return batch

    def __len__(self):
        return self.size


class QMIXAgent:    
    def __init__(self, n_agents, observation_shape, n_actions, learning_rate=5e-4, gamma=0.99, device='cpu'):
        # hyperparams
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        
        # global state dimension
        self.state_dim = n_agents * np.prod(observation_shape)
        
        # equip networks
        self.agent_network = AgentQNetwork(observation_shape, n_actions).to(device)
        self.target_agent_network = AgentQNetwork(observation_shape, n_actions).to(device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        
        self.mixer = QMixNetwork(n_agents, self.state_dim).to(device)
        self.target_mixer = QMixNetwork(n_agents, self.state_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # optimize both agent network and mixer
        params = list(self.agent_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # buffer
        self.buffer = QMIXReplayBuffer(capacity=10000, n_agents=n_agents, 
                                       obs_shape=observation_shape, device=device)

    def select_actions(self, observations_dict, epsilon=0.1):
        actions = {}
        
        for agent, obs in observations_dict.items():
            if random.random() < epsilon:
                actions[agent] = random.randint(0, self.n_actions - 1)
            else:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().to(self.device)
                    q_values = self.agent_network(obs_tensor)
                    actions[agent] = q_values.argmax().item()
        
        return actions
    
    # update network from buffer sample data when enough for a batch.
    def update_q(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0
        
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return 0.0
        
        batch_obs = batch['obs']
        batch_actions = batch['actions']
        batch_rewards = batch['rewards']
        batch_next_obs = batch['next_obs']
        batch_dones = batch['done']
        batch_global_state = batch['global_state']
        batch_next_global_state = batch['next_global_state']
        
        # *get Q-values for chosen actions for each agent*
        agent_qs = []
        for i in range(self.n_agents):
            q_vals = self.agent_network(batch_obs[:, i])
            chosen_q = q_vals.gather(1, batch_actions[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(chosen_q)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        
        # *mix agent Q-values to get total Q*
        chosen_q_tot = self.mixer(agent_qs, batch_global_state)
        
        # target Q-values
        with torch.no_grad():
            target_agent_qs = []
            for i in range(self.n_agents):
                target_q_vals = self.target_agent_network(batch_next_obs[:, i])
                max_q = target_q_vals.max(1)[0]
                target_agent_qs.append(max_q)
            
            target_agent_qs = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer(target_agent_qs, batch_next_global_state)
            
            # *mean reward across agents*
            mean_reward = batch_rewards.mean(dim=1, keepdim=True)
            mean_done = batch_dones.max(dim=1, keepdim=True)[0]
            
            target_q = mean_reward + (1 - mean_done) * self.gamma * target_q_tot
        
        # mean squared loss
        loss = nn.MSELoss()(chosen_q_tot, target_q.detach())
        
        # optim
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_network.parameters()) + 
                                       list(self.mixer.parameters()), 10)
        self.optimizer.step()
        
        return loss.item()
    
    # update target network from q network weights
    def update_target(self):
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())