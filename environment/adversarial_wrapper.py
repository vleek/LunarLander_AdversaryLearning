import os
# --- CRITICAL FIX: STOP THREAD EXPLOSION ---
# This forces Numpy/PyTorch to use 1 core per process.
# Otherwise 126 processes * 128 threads = Server Crash.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------

import gymnasium as gym
import multiprocessing
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
ROUNDS = 10
STEPS_PER_ROUND = 1_000_000  # We can do 1M steps easily now
MODELS_DIR = "models_parallel"
LOG_DIR = "logs_parallel"
MAX_WIND_FORCE = 10.0
VISIBLE_WIND = False

# Don't use all 128 cores. The overhead of managing them outweighs the benefit.
# 32 or 64 is often the "Sweet Spot" for LunarLander.
NUM_CORES = 64  

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make("LunarLander-v3", render_mode=None)
        env = Monitor(env, os.path.join(LOG_DIR, str(rank)))
        env = AdversarialLanderWrapper(
            env, 
            max_wind_force=MAX_WIND_FORCE, 
            visible_wind=VISIBLE_WIND
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print(f"Detected {multiprocessing.cpu_count()} CPUs.")
    print(f"Limiting to {NUM_CORES} cores to minimize overhead.")

    # 1. Create Environment
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CORES)])

    # 2. Initialize Agents with OPTIMIZED Hyperparameters
    print("Initializing Agents...")
    
    # Hyperparameter Tuning for Mass Parallelism:
    # n_steps=128: Collect fewer steps per core because we have many cores.
    #   Batch size = 128 * 64 cores = 8192 (Perfect size)
    # n_epochs=5: Don't over-train on old data, keeps the loop fast.
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "tensorboard_log": LOG_DIR,
        "device": "cpu",
        "n_steps": 256,      # Lower this! (Default is 2048)
        "batch_size": 2048,  # Size of mini-batch for gradient descent
        "n_epochs": 5,       # Spend less time optimizing
        "ent_coef": 0.01     # Encourage exploration
    }

    env.env_method("set_mode", "protagonist")
    protagonist = PPO(**ppo_kwargs)
    
    env.env_method("set_mode", "adversary")
    adversary = PPO(**ppo_kwargs)

    print("Starting Optimized Training...")

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx} / {ROUNDS} {'='*20}")
        
        current_budget = min(20.0 * round_idx, 100.0)
        env.set_attr("max_budget", current_budget)
        
        # --- PROTAGONIST ---
        print(f"> Training Protagonist...")
        adversary.save("temp_adversary_for_workers")
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", "temp_adversary_for_workers.zip")
        
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Protagonist")
        protagonist.save(f"{MODELS_DIR}/protagonist_round_{round_idx}")

        # --- ADVERSARY ---
        print(f"> Training Adversary...")
        protagonist.save("temp_protagonist_for_workers")
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", "temp_protagonist_for_workers.zip")
        
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Adversary")
        adversary.save(f"{MODELS_DIR}/adversary_round_{round_idx}")

    env.close()

if __name__ == "__main__":
    main()