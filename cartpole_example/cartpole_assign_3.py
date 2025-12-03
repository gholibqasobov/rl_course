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
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_size = 4 
action_size = 2  
lr = 0.0001

# Hyperparam for bc
batch_size = 64
expert_trajectories = 1000
max_steps_per_trajectory = 500
num_epochs = 50


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

def potential_function(state):
    return state[0]

def compute_shaped_reward(state, next_state, original_reward, gamma=0.99):
    current_potential = potential_function(state)
    next_potential = potential_function(next_state)
    shaping_reward = gamma * next_potential - current_potential
    shaped_reward = original_reward + shaping_reward
    return shaped_reward, shaping_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train_actor_critic_expert(n_iters=500, use_shaping=True, render=False):
    """Train an expert actor-critic agent"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    
    episode_returns = []
    
    for iter in range(n_iters):
        state, info = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
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
                total_return = sum([r.item() for r in rewards])
                episode_returns.append(total_return)
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
    return actor, critic, episode_returns

def collect_expert_trajectories(actor, num_trajectories=expert_trajectories, max_steps=max_steps_per_trajectory):
    env = gym.make("CartPole-v1")
    expert_states = []
    expert_actions = []
    expert_rewards = []
    
    print(f"Collecting {num_trajectories} trajectories...")
    
    for episode in range(num_trajectories):
        state, info = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for step in range(max_steps):
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state_array).to(device)
            
            with torch.no_grad():
                dist = actor(state_tensor)
                action = dist.sample()
                action_prob = dist.probs
            
            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action.cpu().numpy())
            episode_rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        expert_states.extend(episode_states)
        expert_actions.extend(episode_actions)
        expert_rewards.extend(episode_rewards)
        
        if (episode + 1) % 100 == 0:
            print(f"Collected {episode + 1} trajectories")
    
    env.close()
    
    expert_states = np.array(expert_states, dtype=np.float32)
    expert_actions = np.array(expert_actions, dtype=np.int64)
    expert_rewards = np.array(expert_rewards, dtype=np.float32)
    
    print(f"Total collected: {len(expert_states)} state-action pairs")
    print(f"Average reward per step: {np.mean(expert_rewards):.3f}")
    
    return expert_states, expert_actions

# BC Policy networrk
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(BehaviorCloningPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)
        
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        logits = self.linear3(output)
        return logits

def train_behavior_cloning(expert_states, expert_actions):

    policy = BehaviorCloningPolicy(state_size, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    

    states_tensor = torch.FloatTensor(expert_states).to(device)
    actions_tensor = torch.LongTensor(expert_actions).to(device)
    
    dataset_size = len(expert_states)
    indices = list(range(dataset_size))
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        random.shuffle(indices)
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            
            optimizer.zero_grad()
            logits = policy(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_actions).sum().item()
            total += len(batch_actions)
        
        epoch_loss /= (dataset_size / batch_size)
        accuracy = correct / total
        
        losses.append(epoch_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print(f"Final Training Accuracy: {accuracies[-1]:.4f}")
    
    return policy, losses, accuracies

def evaluate_behavior_cloning_policy(policy, num_episodes=100, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    total_rewards = []
    
    print("\n BC Policy Eval:")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in count():
            if render and episode == 0:
                env.render()
            
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state_array).to(device)
            
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(logits).cpu().numpy()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                total_rewards.append(episode_reward)
                break
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    max_reward = np.max(total_rewards)
    
    print(f"Results:")
    print(f"average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"max reward: {max_reward}")
    print(f"num of epoisodes: {num_episodes}")
    
    return total_rewards, avg_reward, std_reward

def plot_training_results(expert_returns, bc_losses):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Plot 1: Expert training returns
    axes[0].plot(expert_returns)
    axes[0].set_title('Expert Actor-Critic Training Returns')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Behavior cloning loss
    axes[1].plot(bc_losses)
    axes[1].set_title('Behavior Cloning Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    plt.savefig('behavior_cloning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_behavior_cloning_pipeline():
    expert_actor, expert_critic, expert_returns = train_actor_critic_expert(
        n_iters=500, use_shaping=True, render=False
    )

    expert_states, expert_actions = collect_expert_trajectories(
        expert_actor, num_trajectories=expert_trajectories
    )

    bc_policy, bc_losses, bc_accuracies = train_behavior_cloning(expert_states, expert_actions)
    

    bc_eval_rewards, avg_reward, std_reward = evaluate_behavior_cloning_policy(
        bc_policy, num_episodes=100, render=False
    )
    
    # Step 5: Plot results

    plot_training_results(expert_returns, bc_losses)
    
    # Save  trained models
    torch.save({
        'expert_actor_state_dict': expert_actor.state_dict(),
        'expert_critic_state_dict': expert_critic.state_dict(),
        'bc_policy_state_dict': bc_policy.state_dict(),
    }, 'behavior_cloning_models.pth')
    
    return bc_policy, expert_actor, bc_eval_rewards, expert_returns

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    
    bc_policy, expert_actor, bc_eval_rewards, expert_returns = run_behavior_cloning_pipeline()
    
    demo_rewards, _, _ = evaluate_behavior_cloning_policy(bc_policy, num_episodes=3, render=True)
    print(f"Demo rewards: {demo_rewards}")