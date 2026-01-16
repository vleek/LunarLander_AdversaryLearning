import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path so we can import from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Config ---
ROUND_NUM = 10 
# Paths are now relative to the project root
OUTPUT_CSV = "results/evaluation_results.csv"
OUTPUT_IMG = "results/results_matrix_2x2.png"

# Statistical Significance Setting
EPISODES_PER_POINT = 100 

TEST_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315] 
TEST_FORCES = [0.0, 5.0, 10.0, 15.0] 

# Updated paths to point to 'checkpoints/'
MODELS_TO_TEST = [
    {"name": "PPO (Visible)",   "path": f"checkpoints/models_parallel/PPO_Visible/protagonist/round_{ROUND_NUM}",   "type": "PPO",  "visible": True, "row": 0, "col": 0},
    {"name": "LSTM (Visible)",  "path": f"checkpoints/models_parallel/LSTM_Visible/protagonist/round_{ROUND_NUM}",  "type": "LSTM", "visible": True, "row": 0, "col": 1},
    {"name": "PPO (Latent)",    "path": f"checkpoints/models_parallel/PPO_Latent/protagonist/round_{ROUND_NUM}",    "type": "PPO",  "visible": False,"row": 1, "col": 0},
    {"name": "LSTM (Latent)",   "path": f"checkpoints/models_parallel/LSTM_Latent/protagonist/round_{ROUND_NUM}",   "type": "LSTM", "visible": False,"row": 1, "col": 1},
]

class FixedAdversary:
    """Stochastic adversary that targets a specific wind vector with realistic noise."""
    def __init__(self, wind_x, wind_y, max_force=15.0, noise_std=0.1):
        self.target = np.array([wind_x / max_force, wind_y / max_force], dtype=np.float32)
        self.noise_std = noise_std

    def predict(self, obs, deterministic=True):
        noise = np.random.normal(0, self.noise_std, size=self.target.shape)
        action = np.clip(self.target + noise, -1.0, 1.0)
        return action, None

def run_evaluation():
    results = []
    print(f"Starting Evaluation (N={EPISODES_PER_POINT} per point)...")

    for cfg in MODELS_TO_TEST:
        path = cfg["path"]
        # Check for .zip extension handling
        if not os.path.exists(path) and not os.path.exists(path + ".zip"):
            print(f"⚠️ Skipping {cfg['name']} (Not found at {path})")
            continue

        print(f"\nTesting Model: {cfg['name']}")
        
        # Setup Env
        env = gym.make("LunarLander-v3", render_mode=None)
        env = AdversarialLanderWrapper(env, max_wind_force=15.0, visible_wind=cfg['visible'])
        env.max_budget = 100000.0 # Infinite fuel for rigorous testing
        
        # Load Agent
        Loader = RecurrentPPO if cfg["type"] == "LSTM" else PPO
        agent = Loader.load(path, device="cpu")
        
        env.set_mode("protagonist")

        # Grid Search
        for angle in TEST_ANGLES:
            for force in TEST_FORCES:
                # Calc Wind Vector
                rad = np.radians(angle)
                wind_x, wind_y = force * np.cos(rad), force * np.sin(rad)
                
                env.set_opponent(FixedAdversary(wind_x, wind_y))
                print(f" > Angle: {angle:>3}° | Force: {force:>4.1f}...", end="\r")

                success_count = 0
                for _ in range(EPISODES_PER_POINT):
                    obs, _ = env.reset()
                    done = False
                    total_reward = 0
                    
                    lstm_states = None
                    episode_starts = np.ones((1,), dtype=bool)

                    while not done:
                        if cfg["type"] == "LSTM":
                            action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = agent.predict(obs, deterministic=True)
                        
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        done = terminated or truncated

                    if total_reward >= 100: success_count += 1
                
                results.append({
                    "Model": cfg["name"],
                    "Row": cfg["row"],
                    "Col": cfg["col"],
                    "Angle": angle,
                    "Force": force,
                    "Success": success_count / EPISODES_PER_POINT
                })

    df = pd.DataFrame(results)
    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved data to {OUTPUT_CSV}")
    return df

def plot_2x2_matrix(df):
    print("Plotting Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4]) 

    for cfg in MODELS_TO_TEST:
        ax = axes[cfg["row"], cfg["col"]]
        subset = df[df["Model"] == cfg["name"]]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        heatmap_data = subset.pivot_table(index="Force", columns="Angle", values="Success")
        heatmap_data = heatmap_data.sort_index(ascending=False)

        sns.heatmap(
            heatmap_data, ax=ax, cmap="RdYlGn", vmin=0, vmax=1, 
            annot=True, fmt=".2f", cbar=(cfg["row"]==0 and cfg["col"]==0),
            cbar_ax=cbar_ax if (cfg["row"]==0 and cfg["col"]==0) else None
        )
        
        ax.set_title(cfg["name"], fontsize=14, fontweight='bold')
        ax.set_xlabel("Wind Angle (°)")
        ax.set_ylabel("Wind Force")

    fig.suptitle('Robustness Evaluation: Success Rate under Attack', fontsize=20)
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {OUTPUT_IMG}")
    # plt.show() # Commented out for server environments

if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading {OUTPUT_CSV}...")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = run_evaluation()
    plot_2x2_matrix(df)