import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load the Environment (Make sure render_mode is "human" so you can see it)
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Load the Trained Model
# Make sure the path matches where you saved it!
model_path = "models/PPO/ppo_baseline_100k"
model = PPO.load(model_path)

# 3. Enjoy the show
episodes = 5

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    score = 0
    
    while not done:
        # The model predicts the action based on the observation
        # deterministic=True means "do the best move you know" (no randomness)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # Check if the episode is over
        done = terminated or truncated

    print(f"Episode {ep + 1}: Score = {score}")

env.close()