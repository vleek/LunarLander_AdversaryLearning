import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
# Assumes the folder structure created by the automated trainer
# models_parallel/{SCENARIO}/protagonist/round_{ROUND}
ROUND_NUM = 10 
OUTPUT_CSV = "evaluation_results.csv"
OUTPUT_IMG = "results_matrix_2x2.png"

# The 4 Models for the 2x2 Matrix
MODELS_TO_TEST = [
    # Row 1: Visible Wind
    {"name": "PPO (Visible)",   "path": f"models_parallel/PPO_Visible/protagonist/round_{ROUND_NUM}",   "type": "PPO",  "visible": True, "row": 0, "col": 0},
    {"name": "LSTM (Visible)",  "path": f"models_parallel/LSTM_Visible/protagonist/round_{ROUND_NUM}",  "type": "LSTM", "visible": True, "row": 0, "col": 1},
    
    # Row 2: Latent Wind (Invisible)
    {"name": "PPO (Latent)",    "path": f"models_parallel/PPO_Latent/protagonist/round_{ROUND_NUM}",    "type": "PPO",  "visible": False,"row": 1, "col": 0},
    {"name": "LSTM (Latent)",   "path": f"models_parallel/LSTM_Latent/protagonist/round_{ROUND_NUM}",   "type": "LSTM", "visible": False,"row": 1, "col": 1},
]

# Exam Settings
EPISODES_PER_POINT = 5  # Higher = smoother map (try 5 or 10)
TEST_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315] 
TEST_FORCES = [0.0, 5.0, 10.0, 15.0]  # Ramped up to match your 15.0 max

# --- HELPER: FIXED ADVERSARY ---
class FixedAdversary:
    def __init__(self, wind_x, wind_y, max_force=15.0):
        # Normalize force to [-1, 1] range for the wrapper
        self.action = np.array([wind_x / max_force, wind_y / max_force], dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None

def run_evaluation():
    results = []
    print(f"Starting Evaluation for Round {ROUND_NUM} models...")

    for cfg in MODELS_TO_TEST:
        full_path = cfg["path"]
        if not os.path.exists(full_path + ".zip"):
            print(f"⚠️ SKIPPING {cfg['name']}: Model file not found at {full_path}")
            continue

        print(f"\n--- Testing: {cfg['name']} ---")
        
        # 1. Setup Environment
        env = gym.make("LunarLander-v3", render_mode=None)
        env = AdversarialLanderWrapper(env, max_wind_force=15.0, visible_wind=cfg['visible'])
        
        # 2. Load Model
        if cfg["type"] == "LSTM":
            agent = RecurrentPPO.load(full_path, device="cpu")
        else:
            agent = PPO.load(full_path, device="cpu")

        env.set_mode("protagonist")

        # 3. Grid Search
        for angle in TEST_ANGLES:
            for force in TEST_FORCES:
                # Calculate Wind Vector
                rad = np.radians(angle)
                wind_x = force * np.cos(rad)
                wind_y = force * np.sin(rad)
                
                # Force the wind
                env.set_opponent(FixedAdversary(wind_x, wind_y, max_force=15.0))
                
                print(f" > Wind: {angle:>3}° | Force: {force:>4.1f}...", end="\r")

                for i in range(EPISODES_PER_POINT):
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                    
                    # LSTM State Handling
                    lstm_states = None
                    episode_starts = np.ones((1,), dtype=bool)

                    while not done:
                        if cfg["type"] == "LSTM":
                            action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = agent.predict(obs, deterministic=True)
                        
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        done = terminated or truncated

                    # Log Result
                    is_crash = total_reward < 100
                    results.append({
                        "Model": cfg["name"],
                        "Row": cfg["row"],
                        "Col": cfg["col"],
                        "Angle": angle,
                        "Force": force,
                        "Success": 0 if is_crash else 1
                    })

    # Save Data
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nData saved to {OUTPUT_CSV}")
    return df

def plot_2x2_matrix(df):
    print("Generating 2x2 Heatmap Matrix...")
    
    # Setup the 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4]) # Shared colorbar on the right

    # Loop through the 4 subplots
    for cfg in MODELS_TO_TEST:
        row, col = cfg["row"], cfg["col"]
        ax = axes[row, col]
        
        # Filter data for this specific model
        subset = df[df["Model"] == cfg["name"]]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        # Pivot: Rows=Force, Cols=Angle, Values=Mean Success Rate
        heatmap_data = subset.pivot_table(
            index="Force", 
            columns="Angle", 
            values="Success", 
            aggfunc="mean"
        )
        
        # Sort index descending so High Force is at the top
        heatmap_data = heatmap_data.sort_index(ascending=False)

        # Plot Heatmap
        sns.heatmap(
            heatmap_data, 
            ax=ax, 
            cmap="RdYlGn",   # Red=Crash, Green=Safe
            vmin=0, vmax=1,  # Fixed scale 0% to 100%
            annot=True,      # Show numbers
            fmt=".1f", 
            cbar=(row == 0 and col == 0), # Hack to generate one cbar then move it
            cbar_ax=cbar_ax if (row == 0 and col == 0) else None
        )
        
        ax.set_title(cfg["name"], fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Wind Angle (°)")
        ax.set_ylabel("Wind Force")

    # Titles for Rows and Columns (Optional but nice)
    fig.text(0.5, 0.96, 'Robustness Evaluation: Success Rate under Attack', ha='center', fontsize=22)
    
    # Save
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    # 1. Run or Load Data
    if not os.path.exists(OUTPUT_CSV):
        df = run_evaluation()
    else:
        # If you want to force re-run, delete the CSV manually
        print(f"Found existing {OUTPUT_CSV}, plotting directly...")
        df = pd.read_csv(OUTPUT_CSV)

    # 2. Plot
    plot_2x2_matrix(df)