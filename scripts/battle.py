import gymnasium as gym
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Config ---
PROTAGONIST_PATH = "checkpoints/models_parallel/PPO_Visible/protagonist/round_20.zip"
ADVERSARY_PATH   = "checkpoints/models_parallel/PPO_Visible/adversary/round_20.zip"

NO_ADVERSARY     = False
PLOT_WIND        = True
MAX_WIND         = 10.0 # Updated to 10N
EPISODES         = 5

def load_agent(path):
    if not os.path.exists(path) and not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"Loading: {path}")
    if "SAC" in path: return SAC.load(path)
    if "LSTM" in path: return RecurrentPPO.load(path)
    return PPO.load(path)

def run_battle():
    # 1. Detect Settings
    is_visible = "Visible" in PROTAGONIST_PATH
    is_sac = "SAC" in PROTAGONIST_PATH
    use_continuous = True if is_sac else False

    mode_str = f"{'VISIBLE' if is_visible else 'LATENT'} | {'CONTINUOUS' if use_continuous else 'DISCRETE'}"
    print(f"--- Battle Arena Initialized ({mode_str}) ---")

    # 2. Setup Env
    env = gym.make("LunarLander-v3", continuous=use_continuous, render_mode="human")
    env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND, visible_wind=is_visible)

    # 3. Load Agents
    pilot = load_agent(PROTAGONIST_PATH)
    
    if not NO_ADVERSARY:
        wind_god = load_agent(ADVERSARY_PATH)
        env.set_mode("protagonist")
        env.set_opponent(wind_god)
    else:
        print("Adversary Disabled.")
        env.set_mode("protagonist")
        env.set_opponent(None)

    # 4. Plotting
    if PLOT_WIND:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        wind_x, wind_y = [0]*100, [0]*100
        line_x, = ax.plot(wind_x, 'r-', label='Wind X')
        line_y, = ax.plot(wind_y, 'b-', label='Wind Y')
        ax.set_ylim(-MAX_WIND-2, MAX_WIND+2)
        ax.legend()
        ax.set_title("Live Wind Force")

    # 5. Battle Loop
    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        score = 0
        
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        print(f"\nEpisode {ep + 1}/{EPISODES}...")
        
        while not done:
            if isinstance(pilot, RecurrentPPO):
                action, lstm_states = pilot.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = pilot.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
            
            # Update Plot
            if PLOT_WIND:
                w = getattr(env, "current_wind", [0, 0])
                wind_x.append(w[0]); wind_x.pop(0)
                wind_y.append(w[1]); wind_y.pop(0)
                line_x.set_ydata(wind_x)
                line_y.set_ydata(wind_y)
                fig.canvas.draw()
                fig.canvas.flush_events()

            # --- HUD UPDATE ---
            w = getattr(env, "current_wind", [0, 0])
            budget = info.get("budget", 0.0)
            height = obs[1] # Index 1 is always Height (Y-Pos) in LunarLander
            
            # Added Height: {height:5.2f}
            sys.stdout.write(f"\rScore: {score:6.1f} | Budget: {budget:5.1f} | Height: {height:5.2f} | Wind: [{w[0]:5.1f}, {w[1]:5.1f}]")
            sys.stdout.flush()
            time.sleep(0.02) 

        status = "LANDED" if score > 100 else "CRASHED"
        print(f"\n>>> Result: {score:.2f} ({status})")
        time.sleep(1.0)

    env.close()
    if PLOT_WIND: plt.close()

if __name__ == "__main__":
    run_battle()