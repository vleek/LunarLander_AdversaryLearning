import gymnasium as gym
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO  # Needed if you use LSTM
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
# List the paths to the models you want to test
MODELS_TO_TEST = [
    {"name": "PPO_Visible", "path": "models/protagonist_round_10", "type": "PPO", "visible": False},
    {"name": "PPO_Latent",  "path": "models/protagonist_round_5", "type": "PPO", "visible": False},
    # Add your LSTM or SAC models here when they are trained:
    # {"name": "LSTM_Latent", "path": "models_lstm/protagonist_round_10", "type": "LSTM", "visible": False},
]

OUTPUT_FILE = "evaluation_results.csv"
EPISODES_PER_POINT = 5  # How many times to repeat each wind condition (Higher = smoother heatmap)

# Define the "Kill Zone" Grid
TEST_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # Degrees
TEST_FORCES = [0.0, 2.5, 5.0, 7.5, 10.0]            # Wind strength

# --- HELPER: FIXED ADVERSARY ---
# This "Fake Agent" forces the wind to be exactly what we want for the test
class FixedAdversary:
    def __init__(self, wind_x, wind_y):
        # Normalize to -1.0 to 1.0 because the wrapper expects that
        self.action = np.array([wind_x / 10.0, wind_y / 10.0], dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None  # Always return the fixed wind vector

def run_evaluation():
    results = []

    print(f"Starting Evaluation. Saving to {OUTPUT_FILE}...")

    for model_cfg in MODELS_TO_TEST:
        print(f"\n--- Testing Model: {model_cfg['name']} ---")
        
        # 1. Setup Environment
        env = gym.make("LunarLander-v3", render_mode=None)
        env = AdversarialLanderWrapper(env, max_wind_force=10.0, visible_wind=model_cfg['visible'])
        
        # 2. Load Model
        if model_cfg["type"] == "LSTM":
            agent = RecurrentPPO.load(model_cfg["path"], device="cpu")
        else:
            agent = PPO.load(model_cfg["path"], device="cpu")

        env.set_mode("protagonist")

        # 3. Loop through the "Kill Zone" Grid
        for angle in TEST_ANGLES:
            for force in TEST_FORCES:
                # Calculate Wind Vector from Angle/Force
                rad = np.radians(angle)
                wind_x = force * np.cos(rad)
                wind_y = force * np.sin(rad)
                
                # Set the Fixed Adversary
                fake_adversary = FixedAdversary(wind_x, wind_y)
                env.set_opponent(fake_adversary)
                
                print(f" > Testing Wind: {angle}° at Force {force}...", end="\r")

                for i in range(EPISODES_PER_POINT):
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                    fuel_start = info.get("budget", 0) # Just in case we track fuel
                    
                    # LSTM requires state handling
                    lstm_states = None
                    episode_starts = np.ones((1,), dtype=bool)

                    while not done:
                        if model_cfg["type"] == "LSTM":
                            action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = agent.predict(obs, deterministic=True)
                        
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        done = terminated or truncated

                    # 4. Log the Data
                    is_crash = total_reward < 100
                    status = "Crash" if is_crash else "Land"
                    
                    log_entry = {
                        "Model_Name": model_cfg["name"],
                        "Model_Type": model_cfg["type"],
                        "Visibility": "Visible" if model_cfg["visible"] else "Latent",
                        "Wind_Angle": angle,
                        "Wind_Force": force,
                        "Wind_X": wind_x,
                        "Wind_Y": wind_y,
                        "Result_Status": status,
                        "Success": 0 if is_crash else 1,
                        "Total_Reward": total_reward,
                        "Episode_ID": i
                    }
                    results.append(log_entry)

    # 5. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\nDone! Data saved to {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    run_evaluation()