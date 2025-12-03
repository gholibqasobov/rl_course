import minari
import gymnasium as gym
from minari import DataCollector

# Create env
env = gym.make('CartPole-v1')
env = DataCollector(env, record_infos=True)

total_episodes = 100
dataset_id = "cartpole/test-v0"


if dataset_id in minari.list_local_datasets():
    print("Loading local dataset")
    dataset = minari.load_dataset(dataset_id)
else:
    try:
        dataset = minari.load_dataset(dataset_id, download=True)
        print("Download successful.")
    except Exception:
        print("No dataset available")
        dataset = None


for episode_id in range(total_episodes):
    env.reset()
    while True:
        # random policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    if (episode_id + 1) % 10 == 0:
        if dataset is None:
            print(f"Creating dataset {dataset_id}...")
            dataset = env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="Random-Policy",
                description="Random policy CartPole-v1",
                eval_env=env.unwrapped.spec,      
                code_permalink="https://github.com/Farama-Foundation/Minari",
                author="Farama",
                author_email="contact@farama.org"
            )
        else:
            print("Adding episodes to existing dataset...")
            env.add_to_dataset(dataset)
