import gymnasium as gym
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO

# 1. Robust Path Setup
# Get the absolute path to the 'scripts' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project Root is one level up
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Add root to sys.path for imports
sys.path.append(PROJECT_ROOT)

# Define robust paths for artifacts
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "models_parallel")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIG ---
SCENARIO_NAME = "LSTM_Latent" 
TOTAL_ROUNDS = 50
TEST_EPISODES = 20
MAX_WIND = 10.0 # Updated to match your new settings (10N)

class ConstantAdversary:
    def predict(self, obs, deterministic=True):
        # Apply constant horizontal pressure (Force 0.7 * MAX_WIND)
        return np.array([0.7, 0.0], dtype=np.float32), None 

def evaluate_round(round_num):
    # Use the robust path we calculated above
    model_path = os.path.join(CHECKPOINTS_DIR, SCENARIO_NAME, "protagonist", f"round_{round_num}.zip")
    
    if not os.path.exists(model_path):
        # print(f"Round {round_num} not found at: {model_path}") # Debug if needed
        return None

    # Load Agent
    try:
        if "LSTM" in SCENARIO_NAME:
            agent = RecurrentPPO.load(model_path, device="cpu")
        elif "SAC" in SCENARIO_NAME:
            agent = SAC.load(model_path, device="cpu")
        else:
            agent = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Error loading round {round_num}: {e}")
        return None

    # Setup Env
    use_continuous = "SAC" in SCENARIO_NAME
    env = gym.make("LunarLander-v3", continuous=use_continuous, render_mode=None)
    env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND, visible_wind=("Visible" in SCENARIO_NAME))
    
    env.set_mode("protagonist")
    env.set_opponent(ConstantAdversary())
    
    successes = 0
    rewards = []

    for _ in range(TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)

        while not done:
            if "LSTM" in SCENARIO_NAME:
                action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        
        rewards.append(ep_reward)
        if ep_reward > 100: successes += 1

    return np.mean(rewards), (successes / TEST_EPISODES) * 100

def main():
    print(f"Analyzing Progress for: {SCENARIO_NAME}")
    print(f"Looking for models in: {CHECKPOINTS_DIR}")
    data = []

    for r in range(1, TOTAL_ROUNDS + 1):
        print(f" > Evaluating Round {r}/{TOTAL_ROUNDS}...", end="\r")
        result = evaluate_round(r)
        if result:
            avg_score, success_rate = result
            data.append({"Round": r, "Score": avg_score, "Success": success_rate})
    
    if not data:
        print("\n❌ No models found! Check your SCENARIO_NAME and paths.")
        return

    print("\nPlotting...")
    df = pd.DataFrame(data)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Average Score', color=color)
    ax1.plot(df['Round'], df['Score'], color=color, marker='o', label="Score")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Success Rate (%)', color=color)
    ax2.plot(df['Round'], df['Success'], color=color, linestyle='--', marker='x', label="Success %")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    plt.title(f"Training Progress: {SCENARIO_NAME}")
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"progress_{SCENARIO_NAME}.png")
    
    plt.savefig(save_path)
    print(f"✅ Saved plot to: {save_path}")

if __name__ == "__main__":
    main()