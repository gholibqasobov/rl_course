import gymnasium as gym
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode="human").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
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

def trainIters(actor, critic, n_iters, use_shaping=True):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    
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
                if use_shaping:
                    print(f'Iteration: {iter}, Score: {i}, Last Shaping Reward: {shaping_reward:.3f}')
                else:
                    print(f'Iteration: {iter}, Score: {i}')
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
    
    # store models
    os.makedirs('model', exist_ok=True)
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    
    # Train 
    trainIters(actor, critic, n_iters=100, use_shaping=True)