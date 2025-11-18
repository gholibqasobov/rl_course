import gymnasium as gym
import torch
import numpy as np
from itertools import count
import torch.nn.functional as F
from torch.distributions import Categorical
from cartpole_assign_3 import Actor

def visualize_trained_policy(episodes=10):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        actor = torch.load('model/actor.pkl', weights_only=False)
        actor = actor.to(device)  # Move model to the same device
        actor.eval()
        
        env = gym.make("CartPole-v1", render_mode="human").unwrapped
        
        print("Successfully loaded trained policy!")
        
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            
            print(f"Starting Episode {episode + 1}")
            
            for t in count():
                state_array = np.array(state, dtype=np.float32)
                state_tensor = torch.FloatTensor(state_array).to(device)  
                
                # Get action from trained policy
                with torch.no_grad():
                    dist = actor(state_tensor)
                    action = dist.sample()
                
                state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                total_reward += reward
                
                if terminated or truncated:
                    print(f"Episode {episode + 1} finished after {t + 1} timesteps, Total reward: {total_reward}")
                    break
        
        env.close()
        
    except FileNotFoundError:
        print("Error: No trained model found at 'model/actor.pkl'. Please train the model first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    visualize_trained_policy(episodes=5)