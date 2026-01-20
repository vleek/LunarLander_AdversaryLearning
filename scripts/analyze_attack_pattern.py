import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
# Pick your smartest agent (e.g., SAC or LSTM from Round 10)
MODEL_PATH = "checkpoints/models_parallel/LSTM_Visible/adversary/round_10"
ALGO = "LSTM" 
VISIBLE = True

OUTPUT_IMG = "results/attack_pattern_analysis.png"

def run_analysis():
    print(f"Loading Adversary from {MODEL_PATH}...")
    
    # 1. Load Adversary
    if ALGO == "SAC":
        adversary = SAC.load(MODEL_PATH, device="cpu")
    elif ALGO == "LSTM":
        adversary = RecurrentPPO.load(MODEL_PATH, device="cpu")
    else:
        adversary = PPO.load(MODEL_PATH, device="cpu")

    # 2. Setup Env
    # We set max_budget high so we can see their natural behavior without running out
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    env = AdversarialLanderWrapper(env, max_wind_force=15.0, max_budget=200.0, visible_wind=VISIBLE)
    
    # Set to Adversary Mode (No opponent, lander falls naturally to see how wind affects it)
    # OR: Load a protagonist to see a real fight. Let's load a dummy protagonist for now.
    env.set_mode("adversary")
    env.set_opponent(None) # Protagonist does nothing (lander falls straight down)

    obs, _ = env.reset()
    done = False
    
    # Data Collection
    history = []
    step = 0

    print("Running Episode...")
    while not done:
        # Get Action
        action, _ = adversary.predict(obs, deterministic=True)
        
        # Capture State BEFORE step
        # Obs mapping: [0:X, 1:Y, 2:VX, 3:VY, 4:Angle, 5:VA ...]
        # We want Height (Y) and Angle
        # Note: AdversarialWrapper adds dimensions, but original env obs is at start
        raw_obs = obs[:8] 
        height = raw_obs[1]
        angle = raw_obs[4]
        
        # Step
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Capture Wind Applied (from wrapper internals)
        current_wind = np.linalg.norm(env.current_wind)
        
        history.append({
            "Step": step,
            "Height": height,
            "Angle (Tilt)": angle,
            "Wind Force": current_wind
        })
        step += 1

    # 3. Plotting
    df = pd.DataFrame(history)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Angle (Instability) on Left Axis
    sns.lineplot(data=df, x="Step", y="Angle (Tilt)", ax=ax1, color="orange", label="Lander Angle (Instability)")
    ax1.set_ylabel("Lander Angle (Radians)", color="orange")
    ax1.tick_params(axis='y', labelcolor="orange")
    ax1.axhline(0, color='grey', linestyle='--', alpha=0.3)

    # Plot Wind Force on Right Axis
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Step", y="Wind Force", ax=ax2, color="blue", label="Adversary Attack Force", alpha=0.6)
    ax2.set_ylabel("Wind Force applied (Newtons)", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    
    # Fill area to show "Bursts"
    ax2.fill_between(df["Step"], df["Wind Force"], color="blue", alpha=0.1)

    plt.title(f"Adversary Intelligence: Does it attack instability? ({ALGO})", fontsize=14)
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ Analysis saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    run_analysis()