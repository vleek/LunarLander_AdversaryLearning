import os
# --- CRITICAL FIX: STOP THREAD EXPLOSION ---
# This forces Numpy/PyTorch to use 1 core per process.
# Otherwise 64 processes * 64 threads = Server Crash.
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
STEPS_PER_ROUND = 500_000  # Total 10M steps
MODELS_DIR = "models_parallel"
LOG_DIR = "logs_parallel"

# Environment Physics
MAX_WIND_FORCE = 15.0      # Strong enough to force crashes
VISIBLE_WIND = False       # Set False for Latent runs, True for Baseline

# Cores: Set this to your actual CPU count (e.g. 64)
NUM_CORES = 64  

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(rank: int, seed: int = 0):
    def _init():
        # Create base env
        env = gym.make("LunarLander-v3", render_mode=None)
        # Wrap for logging
        env = Monitor(env, os.path.join(LOG_DIR, str(rank)))
        # Wrap for Adversarial Physics
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

    # 1. Create Environment FIRST (Fixes NameError)
    # This spins up 64 separate Python processes
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CORES)])

    # 2. Define Hyperparameters SECOND (Now env exists)
    print("Initializing Agents...")
    
    # Hyperparameter Tuning for Mass Parallelism (64 Cores):
    # n_steps=128: Collect fewer steps per core.
    # Batch size = 128 * 64 cores = 8192 samples per update (Optimal)
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "tensorboard_log": LOG_DIR,
        "device": "cpu",
        
        "n_steps": 128,      # Lower this! (Default is 2048)
        "batch_size": 2048,  # Size of mini-batch for gradient descent
        "n_epochs": 5,       # Spend less time optimizing old data
        "learning_rate": 3e-4, 
        "ent_coef": 0.01,    # Encourage exploration
        "gamma": 0.995,      # Long horizon
    }

    # 3. Initialize Agents
    # Protagonist (The Pilot)
    env.env_method("set_mode", "protagonist")
    protagonist = PPO(**ppo_kwargs)
    
    # Adversary (The Wind God)
    env.env_method("set_mode", "adversary")
    adversary = PPO(**ppo_kwargs)

    print("Starting Optimized Training...")

    # 4. Training Loop
    for round_idx in range(1, ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx} / {ROUNDS} {'='*20}")
        
        # Curriculum: Slowly increase Adversary Budget
        current_budget = min(20.0 * round_idx, 100.0)
        
        # IMPORTANT: Use set_attr for Parallel Envs
        env.set_attr("max_budget", current_budget)
        print(f"Adversary Budget set to: {current_budget}")
        
        # --- PHASE A: TRAIN PROTAGONIST ---
        print(f"> Training Protagonist (vs Frozen Adversary)...")
        
        # Save current adversary state to file so workers can load it
        adversary.save("temp_adversary_for_workers")
        
        # Instruct workers to become Protagonists & load the Enemy
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", "temp_adversary_for_workers.zip")
        
        # Train
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Protagonist")
        protagonist.save(f"{MODELS_DIR}/protagonist_PPO_F_{round_idx}")

        # --- PHASE B: TRAIN ADVERSARY ---
        print(f"> Training Adversary (vs Frozen Protagonist)...")
        
        # Save current protagonist state
        protagonist.save("temp_protagonist_for_workers")
        
        # Instruct workers to become Adversaries & load the Target
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", "temp_protagonist_for_workers.zip")
        
        # Train
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Adversary")
        adversary.save(f"{MODELS_DIR}/adversary_PPO_F_{round_idx}")

    env.close()
    print("Training Complete.")

if __name__ == "__main__":
    main()