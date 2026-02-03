import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import safe_globals
from collections import deque
import os

from pettingzoo.sisl import pursuit_v4
import QMix
import IQL


def load_IQL(model_name, pursuit_config, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    env = pursuit_v4.parallel_env(**pursuit_config)
    agents = env.possible_agents
    obs_shape = env.observation_space(agents[0]).shape
    n_actions = env.action_space(agents[0]).n
    env.close()
    
    marl_agents = {agent: IQL.MARLAgent(obs_shape, n_actions, device=device) for agent in agents}
    
    for agent_name in agents:
        file_path = os.path.join(f"./{model_name}", f"{agent_name}_IQL.pth")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Saved model not found: {file_path}")
        marl_agents[agent_name].q_network.load_state_dict(torch.load(file_path, map_location=device))
        marl_agents[agent_name].update_target()
    
    print(f"Loaded IQL agents from '{model_name}'")
    
    return marl_agents


def load_QMIX(filepath, device='cpu'):
    """
    Load a saved QMIX agent checkpoint.
    """
    import torch
    from torch.serialization import safe_globals

    # safe loading for PyTorch >=2.6
    with safe_globals(['numpy._core.multiarray.scalar']):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    n_agents = checkpoint['n_agents']
    obs_shape = checkpoint['obs_shape']
    n_actions = checkpoint['n_actions']

    # Initialize QMIX agent (match your constructor)
    qmix_agent = QMix.QMIXAgent(n_agents=n_agents, observation_shape=obs_shape, n_actions=n_actions)  # adjust args to your class

    # Load the saved networks
    qmix_agent.agent_network.load_state_dict(checkpoint['agent_network'])
    qmix_agent.mixer.load_state_dict(checkpoint['mixer'])

    print(f"Loaded QMIX agent from '{filepath}'")
    return qmix_agent



def marl_agents_to_policy(marl_agents, epsilon=0.0):
    def policy_fn(agent, observations):
        actions = marl_agents[agent].select_action(observations[agent], epsilon)
        return actions
    return policy_fn


def qmix_agent_to_policy(qmix_agent, epsilon=0.0):
    def policy_fn(agent, observations):
        actions = qmix_agent.select_actions(observations, epsilon=epsilon)
        return actions[agent]
    return policy_fn
