"""
Multi-Agent Reinforcement Learning for PettingZoo Pursuit Environment
This implementation uses Independent Q-Learning (IQL) where each agent learns independently.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pettingzoo.sisl import pursuit_v4
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    """Deep Q-Network for agent policy."""
    
    def __init__(self, obs_shape, n_actions, hidden_dim=128):
        super(QNetwork, self).__init__()
        
        # Calculate input dimension from observation shape
        if len(obs_shape) == 3:  # Image observations
            self.conv = nn.Sequential(
                nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate flattened size
            conv_out_size = obs_shape[0] * obs_shape[1] * 64
        else:
            self.conv = nn.Flatten()
            conv_out_size = np.prod(obs_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        #if len(x.shape) == 3:  # Single observation
        #x = x.unsqueeze(0)
        
        # Permute to match PyTorch's expected format (batch, channels, height, width)
        #if len(x.shape) == 4 and x.shape[-1] <= 3:
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        return self.fc(x)


class ReplayBuffer:
    """Experience replay buffer for training."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
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
    """Independent Q-Learning Agent."""
    
    def __init__(self, obs_shape, n_actions, learning_rate=1e-3, gamma=0.99, device='cpu'):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        
        # Q-networks
        self.q_network = QNetwork(obs_shape, n_actions).to(device)
        self.target_network = QNetwork(obs_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size=32):
        """Update Q-network using experience replay."""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_marl(n_episodes=1000, max_steps=500, batch_size=32, 
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
               target_update_freq=10):
    """Train MARL agents on Pursuit environment."""
    
    # Initialize environment
    env = pursuit_v4.parallel_env(max_cycles=max_steps, x_size=16, y_size=16,
                                   n_evaders=30, n_pursuers=8)
    env.reset()
    
    # Get observation and action spaces
    agents = env.possible_agents
    obs_shape = env.observation_space(agents[0]).shape
    n_actions = env.action_space(agents[0]).n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Number of agents: {len(agents)}")
    
    # Initialize agents
    marl_agents = {agent: MARLAgent(obs_shape, n_actions, device=device) 
                   for agent in agents}
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    epsilon = epsilon_start
    
    print("\nStarting training...")
    for episode in range(n_episodes):
        observations, infos = env.reset()
        episode_reward = {agent: 0 for agent in agents}
        step_count = 0
        episode_losses = []
        
        while env.agents:
            # Select actions for all agents
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    actions[agent] = marl_agents[agent].select_action(
                        observations[agent], epsilon
                    )
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions
            for agent in actions.keys():
                if agent in observations and agent in next_observations:
                    marl_agents[agent].replay_buffer.push(
                        observations[agent],
                        actions[agent],
                        rewards[agent],
                        next_observations[agent],
                        terminations[agent] or truncations[agent]
                    )
                    episode_reward[agent] += rewards[agent]
            
            # Update networks
            for agent in agents:
                loss = marl_agents[agent].update(batch_size)
                if loss > 0:
                    episode_losses.append(loss)
            
            observations = next_observations
            step_count += 1
        
        # Update target networks
        if episode % target_update_freq == 0:
            for agent in agents:
                marl_agents[agent].update_target_network()
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record metrics
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(step_count)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            recent_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {recent_reward:.2f} | "
                  f"Avg Length: {recent_length:.1f} | "
                  f"Epsilon: {epsilon:.3f}")
    
    env.close()
    
    return marl_agents, episode_rewards, episode_lengths, losses


def evaluate_marl(marl_agents, n_episodes=10, max_steps=500, render=False):
    """Evaluate trained MARL agents."""
    
    env = pursuit_v4.parallel_env(max_cycles=max_steps, x_size=16, y_size=16,
                                   n_evaders=30, n_pursuers=8)
    
    episode_rewards = []
    
    print("\nEvaluating trained agents...")
    for episode in range(n_episodes):
        observations, infos = env.reset()
        episode_reward = {agent: 0 for agent in marl_agents.keys()}
        
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    actions[agent] = marl_agents[agent].select_action(
                        observations[agent], epsilon=0.0  # Greedy
                    )
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent in actions.keys():
                if agent in rewards:
                    episode_reward[agent] += rewards[agent]
        
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}")
    
    env.close()
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")
    
    return episode_rewards


def plot_training_results(episode_rewards, episode_lengths, losses, save_path=None):
    """Plot training metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6)
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 
                    linewidth=2, label='Moving Avg (50)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6)
    axes[0, 1].plot(np.convolve(episode_lengths, np.ones(50)/50, mode='valid'),
                    linewidth=2, label='Moving Avg (50)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.6)
        if len(losses) > 50:
            axes[1, 0].plot(np.convolve(losses, np.ones(50)/50, mode='valid'),
                           linewidth=2, label='Moving Avg (50)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot reward distribution
    axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Average Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Train agents
    print("="*60)
    print("MARL Training for PettingZoo Pursuit Environment")
    print("="*60)
    
    marl_agents, rewards, lengths, losses = train_marl(
        n_episodes=10,
        max_steps=500,
        batch_size=64,
        epsilon_decay=0.995
    )
    
    # Evaluate agents
    print("\n" + "="*60)
    eval_rewards = evaluate_marl(marl_agents, n_episodes=10)
    
    # Plot results
    plot_training_results(rewards, lengths, losses, 
                         save_path='.')
    
    # Save models
    print("\nSaving trained models...")
    for agent_name, agent in marl_agents.items():
        torch.save(agent.q_network.state_dict(), 
                  f'./marl_agent_{agent_name}.pth')
    
    print("\nTraining complete!")