import gymnasium as gym

# We use v3 as discussed in your proposal
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()