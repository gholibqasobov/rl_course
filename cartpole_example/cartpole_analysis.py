import gymnasium as gym

# Discrete action space (button presses)
env = gym.make("CartPole-v1")

print(f"Action space: {env.action_space}")

print(f"Sample action: {env.action_space.sample()}")


# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values

print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation