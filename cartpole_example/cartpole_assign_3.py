import gymnasium as gym
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = 4  # CartPole observation space size
action_size = 2  # CartPole action space size
lr = 0.0001

def potential_function(state):
    return state[0]

def compute_shaped_reward(state, next_state, original_reward, gamma=0.99):
    current_potential = potential_function(state)
    next_potential = potential_function(next_state)
    shaping_reward = gamma * next_potential - current_potential
    shaped_reward = original_reward + shaping_reward
    return shaped_reward, shaping_reward

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def trainIters(actor, critic, n_iters, use_shaping=True, render=False):
    # Create a new environment for each training run
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    
    episode_returns = []  # Store returns for each episode
    
    for iter in range(n_iters):
        state, info = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        
        # previous state 
        prev_state = state

        for i in count():
            if render:
                env.render()
            
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state_array).to(device)
            
            dist, value = actor(state_tensor), critic(state_tensor)

            action = dist.sample()
            
            next_state, original_reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated 

            # potential-based reward shaping
            if use_shaping and i > 0: 
                shaped_reward, shaping_reward = compute_shaped_reward(
                    prev_state, next_state, original_reward
                )
                reward_to_use = shaped_reward
            else:
                reward_to_use = original_reward
                shaping_reward = 0.0

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward_to_use], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            prev_state = state
            state = next_state

            if done:
                # Calculate total return for this episode
                total_return = sum([r.item() for r in rewards])
                episode_returns.append(total_return)
                
                if use_shaping:
                    print(f'Iteration: {iter}, Score: {i}, Total Return: {total_return:.1f}, Last Shaping Reward: {shaping_reward:.3f}')
                else:
                    print(f'Iteration: {iter}, Score: {i}, Total Return: {total_return:.1f}')
                break

        next_state_array = np.array(next_state, dtype=np.float32)
        next_state_tensor = torch.FloatTensor(next_state_array).to(device)
        next_value = critic(next_state_tensor)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    
    env.close()
    return episode_returns

def plot_comparison(returns_with_shaping, returns_without_shaping, window_size=10):

    plt.figure(figsize=(12, 6))
    
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Plot raw returns with transparency
    # plt.plot(returns_with_shaping, alpha=0.3, color='blue', label='With Shaping (raw)')
    # plt.plot(returns_without_shaping, alpha=0.3, color='red', label='Without Shaping (raw)')
    
    # Plot smoothed returns
    if len(returns_with_shaping) >= window_size:
        smoothed_with = moving_average(returns_with_shaping, window_size)
        plt.plot(range(window_size-1, len(returns_with_shaping)), smoothed_with, 
                color='blue', linewidth=2, label=f'With Shaping (MA{window_size})')
    
    if len(returns_without_shaping) >= window_size:
        smoothed_without = moving_average(returns_without_shaping, window_size)
        plt.plot(range(window_size-1, len(returns_without_shaping)), smoothed_without, 
                color='red', linewidth=2, label=f'Without Shaping (MA{window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Average Return Over Episodes: With vs Without Reward Shaping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    avg_with = np.mean(returns_with_shaping)
    avg_without = np.mean(returns_without_shaping)
    std_with = np.std(returns_with_shaping)
    std_without = np.std(returns_without_shaping)
    
    plt.figtext(0.02, 0.02, 
                f'With Shaping: Mean = {avg_with:.1f} ± {std_with:.1f}\n'
                f'Without Shaping: Mean = {avg_without:.1f} ± {std_without:.1f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('reward_shaping_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_comparison(n_iters=100):
    """
    Run training with and without reward shaping and plot comparison
    """
    print("Training with reward shaping...")
    # Create new models for with shaping
    actor_with = Actor(state_size, action_size).to(device)
    critic_with = Critic(state_size, action_size).to(device)
    returns_with_shaping = trainIters(actor_with, critic_with, n_iters=n_iters, use_shaping=True, render=False)
    
    print("\nTraining without reward shaping...")
    # Create new models for without shaping
    actor_without = Actor(state_size, action_size).to(device)
    critic_without = Critic(state_size, action_size).to(device)
    returns_without_shaping = trainIters(actor_without, critic_without, n_iters=n_iters, use_shaping=False, render=False)
    
    # Plot comparison
    plot_comparison(returns_with_shaping, returns_without_shaping)
    
    return returns_with_shaping, returns_without_shaping

# Alternative: Run single training (comment out run_comparison if using this)
def run_single_training(use_shaping=True, n_iters=100, render=False):
    """Run a single training session with or without reward shaping"""
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    
    returns = trainIters(actor, critic, n_iters=n_iters, use_shaping=use_shaping, render=render)
    
    # Plot single training results
    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    shaping_text = "with" if use_shaping else "without"
    plt.title(f'Training Returns {shaping_text} Reward Shaping')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'training_returns_{shaping_text}_shaping.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return returns

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    
    # Option 1: Run comparison between with and without reward shaping
    returns_with, returns_without = run_comparison(n_iters=1000)
    
    # Option 2: Run single training (uncomment below and comment the above line)
    # returns = run_single_training(use_shaping=True, n_iters=100, render=False)