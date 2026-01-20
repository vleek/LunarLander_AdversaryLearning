import gymnasium as gym
import numpy as np
import pandas as pd
import os
import sys
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from gymnasium import spaces

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
# Check a spread of rounds to see evolution
ROUNDS_TO_CHECK = [2, 4, 6, 8, 10]  
MAX_WIND_FORCE = 10.0 
TEST_EPISODES = 3

# Define ALL models to check
MODELS = [
    # --- VISIBLE (Inputs include wind vector) ---
    {"name": "PPO_Visible",  "algo": "PPO",  "path_base": "checkpoints/models_parallel/PPO_Visible/adversary"},
    {"name": "LSTM_Visible", "algo": "LSTM", "path_base": "checkpoints/models_parallel/LSTM_Visible/adversary"},
    {"name": "SAC_Visible",  "algo": "SAC",  "path_base": "checkpoints/models_sac_gpu/SAC_Visible/adversary"},

    # --- LATENT (Inputs hidden, must infer wind) ---
    {"name": "PPO_Latent",   "algo": "PPO",  "path_base": "checkpoints/models_parallel/PPO_Latent/adversary"},
    {"name": "LSTM_Latent",  "algo": "LSTM", "path_base": "checkpoints/models_parallel/LSTM_Latent/adversary"},
    {"name": "SAC_Latent",   "algo": "SAC",  "path_base": "checkpoints/models_sac_gpu/SAC_Latent/adversary"},
]

def load_agent(path, algo):
    try:
        if algo == "LSTM": return RecurrentPPO.load(path, device="cpu")
        if algo == "SAC": return SAC.load(path, device="cpu")
        return PPO.load(path, device="cpu")
    except Exception as e:
        return None

def diagnose_model(name, algo, path, round_num):
    full_path = os.path.join(PROJECT_ROOT, f"{path}/round_{round_num}.zip")
    
    if not os.path.exists(full_path):
        return None

    # Load Agent
    agent = load_agent(full_path, algo)
    if not agent:
        return {"Model": name, "Round": round_num, "Status": "❌ LOAD ERROR"}

    # Setup Env
    is_continuous = (algo == "SAC")
    is_visible = "Visible" in name
    
    env = gym.make("LunarLander-v3", continuous=is_continuous, render_mode=None)
    # Use max_budget=100.0 to verify if they respect the standard limit
    env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND_FORCE, max_budget=100.0, visible_wind=is_visible)
    
    # Set to Adversary Mode (No Opponent loaded, just testing wind behavior)
    env.set_mode("adversary") 
    env.set_opponent(None) 

    total_wind_force = []
    budgets_end = []
    
    for _ in range(TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        ep_wind_sum = 0
        
        while not done:
            # Predict
            if algo == "LSTM":
                action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            
            # Step
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record actual wind applied
            current_wind = np.linalg.norm(getattr(env, "current_wind", [0, 0]))
            ep_wind_sum += current_wind
            
        total_wind_force.append(ep_wind_sum)
        budgets_end.append(info.get("budget", 0))

    avg_wind = np.mean(total_wind_force)
    avg_budget_left = np.mean(budgets_end)
    used_budget = 100.0 - avg_budget_left

    # --- DIAGNOSIS LOGIC ---
    status = "✅ HEALTHY"
    
    # 1. LAZY: Uses almost no budget (< 5%)
    if used_budget < 5.0:
        status = "💤 LAZY (No Action)"
        
    # 2. MANIC: Spams massive wind total regardless of budget limits
    elif used_budget > 95.0 and avg_wind > 5000: 
        status = "⚠️ MANIC (Spams Wind)"
        
    # 3. BROKE: Uses up all budget but within normal physics limits
    elif used_budget > 95.0:
        status = "⚠️ BROKE (Ran out of money)"

    return {
        "Model": name,
        "Round": round_num,
        "Avg Wind Pressure": round(avg_wind, 1),
        "Budget Used": round(used_budget, 1),
        "Status": status
    }

def main():
    print("--- DIAGNOSING ALL ADVERSARY AGENTS (VISIBLE & LATENT) ---")
    results = []

    for model_cfg in MODELS:
        print(f"Checking {model_cfg['name']}...")
        for r in ROUNDS_TO_CHECK:
            res = diagnose_model(model_cfg["name"], model_cfg["algo"], model_cfg["path_base"], r)
            if res:
                results.append(res)
                print(f"  > Round {r}: {res['Status']} (Used: {res['Budget Used']}%)")
            else:
                # print(f"  > Round {r}: File not found") # Uncomment to debug missing files
                pass

    # Summary Table
    if results:
        df = pd.DataFrame(results)
        print("\n--- FINAL DIAGNOSIS REPORT ---")
        # Sort by Model Name then Round for easier reading
        df = df.sort_values(by=["Model", "Round"])
        print(df.to_string(index=False))
        
        save_path = os.path.join(PROJECT_ROOT, "results", "diagnosis_report_all.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nSaved full report to {save_path}")

if __name__ == "__main__":
    main()