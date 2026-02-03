import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_reward(reward_history, smooth=10):
    rewards = np.array(reward_history)
    avg_rewards = rewards.mean(axis=1)
    sum_rewards = rewards.sum(axis=1)

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        avg_smooth = np.convolve(avg_rewards, kernel, mode="valid")
        sum_smooth = np.convolve(sum_rewards, kernel, mode="valid")
        x_smooth = np.arange(len(avg_smooth))
    else:
        avg_smooth = avg_rewards
        sum_smooth = sum_rewards
        x_smooth = np.arange(len(avg_rewards))

    x = np.arange(len(avg_rewards))
    plt.figure(figsize=(9, 6))

    plt.plot(x, avg_rewards, alpha=0.3, label="Avg reward (raw)")
    plt.plot(x, sum_rewards, alpha=0.3, label="Sum reward (raw)")

    plt.plot(x_smooth, avg_smooth, linewidth=2,
             label=f"Avg reward (smoothed {smooth})")

    plt.plot(x_smooth, sum_smooth, linewidth=2,
             label=f"Sum reward (smoothed {smooth})")

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Average and Sum Reward per Step")
    plt.legend()
    plt.grid(True)

    plt.show()


def save_data(filename, reward_history, infos=None):
    rewards = np.array(reward_history)
    avg_rewards = rewards.mean(axis=1)
    sum_rewards = rewards.sum(axis=1)

    data = {
        "step": np.arange(len(rewards)),
        "avg_reward": avg_rewards,
        "sum_reward": sum_rewards
    }

    if infos is not None:
        for key, value in infos.items():
            data[key] = value

    df_new = pd.DataFrame(data)

    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        run_id = df_old["run_id"].max() + 1
    else:
        run_id = 0
        df_old = None

    df_new["run_id"] = run_id

    if df_old is not None:
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(filename, index=False)
    print(f"Saved run {run_id} to {filename}")


def plot_training_results(episode_rewards, episode_lengths, losses, save_path=None):    
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
    
    # Save figure if path is given
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_eval_metrics(episode_rewards, episode_lengths, action_counts, success_flags=None, smooth=10, save_path=None):
    n_plots = 3 if success_flags is None else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    elif n_plots == 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Episode Rewards
    rewards = np.array(episode_rewards)
    if smooth > 1:
        rewards_smooth = np.convolve(rewards, np.ones(smooth)/smooth, mode='valid')
        x_smooth = np.arange(len(rewards_smooth))
    else:
        rewards_smooth = rewards
        x_smooth = np.arange(len(rewards))
    
    axes[0].plot(np.arange(len(rewards)), rewards, alpha=0.3, label="Raw Reward")
    axes[0].plot(x_smooth, rewards_smooth, linewidth=2, label=f"Smoothed ({smooth})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Episode Lengths
    lengths = np.array(episode_lengths)
    if smooth > 1:
        lengths_smooth = np.convolve(lengths, np.ones(smooth)/smooth, mode='valid')
        x_smooth = np.arange(len(lengths_smooth))
    else:
        lengths_smooth = lengths
        x_smooth = np.arange(len(lengths))
    
    axes[1].plot(np.arange(len(lengths)), lengths, alpha=0.3, label="Raw Length")
    axes[1].plot(x_smooth, lengths_smooth, linewidth=2, label=f"Smoothed ({smooth})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Lengths")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Action Distribution
    if success_flags is not None:
        ax_actions = fig.add_axes([0.72, 0.1, 0.25, 0.35])  # small inset for action counts
    else:
        ax_actions = axes[2]
    
    actions = np.array(action_counts)
    ax_actions.bar(np.arange(len(actions)), actions, alpha=0.7, edgecolor='black')
    ax_actions.set_xlabel("Action Index")
    ax_actions.set_ylabel("Count")
    ax_actions.set_title("Action Distribution")
    ax_actions.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is given
    if save_path:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()