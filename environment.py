from pettingzoo.sisl import pursuit_v4
import visualization

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import IQL

def start_AEC(configuration, policy=None):
    env = pursuit_v4.env(render_mode="human", **configuration)
    env.reset(seed=42)

    attained_goal = False
    step_count = 0

    reward_history = []
    reward_history_current = []

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination:
            print(f"agent {agent} terminated")
            action = None
            attained_goal = True
        elif truncation:
            print(f"agent {agent} truncated")
            action = None
        else:
            if policy is None:
                action = env.action_space(agent).sample()
            else:
                action = policy(agent, observation)

        env.step(action)
        step_count += 1
        reward_history_current.append(reward)
        if step_count % configuration["n_pursuers"] == 0:
            reward_history.append(reward_history_current)
            reward_history_current = []

    if attained_goal:
        print(f"pursuit successful after {(step_count - 1) // configuration['n_pursuers']} steps!")
    else:
        print(f"pursuit failed, interrupted after {(step_count - 1) // configuration['n_pursuers']} steps.")
    
    visualization.plot_reward(reward_history)
    visualization.save_data("data.csv", reward_history)
    
    env.close()


def start_parallel(configuration, policy=None):
    env = pursuit_v4.parallel_env(render_mode="human", **configuration)
    observations, infos = env.reset()

    attained_goal = False
    step_count = 0

    reward_history = []

    while env.agents:
        if policy is None:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions = {agent: policy(agent, observations[agent]) for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        for agent in terminations:
            if terminations[agent]:
                print(f"agent {agent} terminated")
            if truncations[agent]: 
                print(f"agent {agent} truncated")

        reward_history.append(list(rewards.values()))

    if attained_goal:
        print(f"pursuit successful after {step_count} steps!")
    else:
        print(f"pursuit failed, interrupted after {step_count} steps.")
    
    visualization.plot_reward(reward_history)
    visualization.save_data("data.cvs", reward_history)
    
    env.close()


def train_IQL(pursuit_config,
              model_name=None,
              n_episodes=1000, 
              batch_size=32, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
              target_update_freq=10,
              ):
    # initialize environment
    env = pursuit_v4.parallel_env(**pursuit_config)
    env.reset()

    # initialize model
    agents = env.possible_agents
    obs_shape = env.observation_space(agents[0]).shape
    n_actions = env.action_space(agents[0]).n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Number of agents: {len(agents)}")

    marl_agents = {agent: IQL.MARLAgent(obs_shape, n_actions, device=device) for agent in agents}

    # training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    # training
    print("\nIQL training started...")
    epsilon = epsilon_start
    for episode in range(n_episodes):
        # init
        observations, info = env.reset()
        episode_reward = {agent: 0 for agent in agents}
        step_count = 0
        episode_losses = []

        # all agents take a step
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    actions[agent] = marl_agents[agent].select_action(observations[agent], epsilon)
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

            # save transitions in buffer
            for agent in actions.keys():
                if agent in observations and agent in next_observations:
                    marl_agents[agent].buffer.push(
                        observations[agent],
                        actions[agent],
                        rewards[agent],
                        next_observations[agent],
                        terminations[agent] or truncations[agent]
                    )
                    episode_reward[agent] += rewards[agent]        
        
            # update Q NNs
            for agent in agents:
                loss = marl_agents[agent].update_q(batch_size)
                if loss > 0:
                    episode_losses.append(loss)
        
            observations = next_observations

        # update target NNs
        if episode % target_update_freq == 0:
            for agent in agents:
                marl_agents[agent].update_target()

        # epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # save 'n log
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(step_count)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            recent_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {recent_reward:.2f} | "
                  f"Avg Length: {recent_length:.1f} | "
                  f"Epsilon: {epsilon:.3f}")
        
    env.close()
    print("\nIQL training complete!")

    metadata = {}

    if model_name is not None:
        saves = []
        for agent_name, agent in marl_agents.items():
            file_name = f"./{model_name}_{agent_name}.pth"
            torch.save(agent.q_network.state_dict(), file_name)
            saves.append(file_name)
        metadata["filenames"] = saves

    return marl_agents, episode_rewards, episode_lengths, losses, metadata


def eval_IQL(pursuit_config,
             marl_agents, 
             n_episodes=10, 
             render=False):
    if render:
        pursuit_config["render_mode"] = "human"
    env = pursuit_v4.parallel_env(**pursuit_config)

    episode_rewards = []

    print("\nIQL evaluation started...")
    for episode in range(n_episodes):
        observations, infos = env.reset()

        episode_reward = {agent: 0 for agent in marl_agents.keys()}
        
        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    # epsilon set to 0 for accurate evaluation
                    actions[agent] = marl_agents[agent].select_action(observations[agent], epsilon=0.0)
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent in actions.keys():
                if agent in rewards:
                    episode_reward[agent] += rewards[agent]
        
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}")

        env.close()
    
    print("\nIQL evaluation complete!")
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")
    
    return episode_rewards