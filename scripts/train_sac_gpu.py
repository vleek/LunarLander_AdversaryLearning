import os
import sys
import torch
import gymnasium as gym
import numpy as np
import shutil

# --- 1. PATH SETUP (Critical for importing from 'src') ---
# Add the project root directory to the python path so we can see 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Config ---
ROUNDS = 50
STEPS_PER_ROUND = 100000 
MAX_WIND_FORCE = 10
NUM_ENVS = 4

# Updated paths to point to the new folder structure
# Go up one level ("..") then into "checkpoints" or "logs_sac"
BASE_MODELS = os.path.join("..", "checkpoints", "models_sac_gpu")
BASE_LOGS = os.path.join("..", "logs_sac")

SCENARIOS = [
    {"name": "SAC_Visible", "visible": True},
    {"name": "SAC_Latent",  "visible": False},
]

# Check GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {DEVICE}")

class RobustnessCallback(BaseCallback):
    """Logs CVaR (Safety) and Impact Velocity (Crash Intensity) to CSV."""
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            buf = self.model.ep_info_buffer
            if buf:
                rewards = [i["r"] for i in buf]
                impacts = [i["impact_vel"] for i in buf if "impact_vel" in i]

                # CVaR (Worst 5% of recent episodes)
                if len(rewards) >= 20:
                    worst = np.sort(rewards)[:max(1, int(len(rewards) * 0.05))]
                    self.logger.record("robustness/cvar_5_percent", np.mean(worst))

                # Average Crash Intensity
                if impacts:
                    self.logger.record("robustness/avg_impact_vel", np.mean(impacts))
        return True

def make_env(rank, visible_wind, log_dir, seed=0):
    def _init():
        # 1. Create Base Env
        env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
        
        # 2. Add Adversarial Logic FIRST (This adds 'impact_vel' to info)
        env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND_FORCE, visible_wind=visible_wind)
        
        # 3. Add Monitor LAST (Now it can see 'impact_vel')
        env = Monitor(env, os.path.join(log_dir, str(rank)), info_keywords=("impact_vel",))
        
        env.reset(seed=seed + rank)
        return env
    return _init

def get_sac_config(env):
    return {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "device": DEVICE,
        # Turbo Settings
        "buffer_size": 100_000, 
        "batch_size": 256,
        "train_freq": (200, "step"), 
        "gradient_steps": 50,
        "learning_starts": 1000,
        "learning_rate": 3e-4,
        "ent_coef": "auto",
        "gamma": 0.995,
    }

def run_scenario(scenario):
    name, vis = scenario["name"], scenario["visible"]
    print(f"\n>>> Starting SAC Training: {name}")

    path_model = os.path.join(BASE_MODELS, name)
    path_log = os.path.join(BASE_LOGS, name)
    os.makedirs(f"{path_model}/protagonist", exist_ok=True)
    os.makedirs(f"{path_model}/adversary", exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # Init
    env = SubprocVecEnv([make_env(i, vis, path_log) for i in range(NUM_ENVS)])
    sac_kwargs = get_sac_config(env)
    
    # Agents
    env.env_method("set_mode", "protagonist")
    protagonist = SAC(**sac_kwargs)
    
    # Force CSV logging for Protagonist
    new_logger = configure(path_log, ["stdout", "csv", "tensorboard"])
    protagonist.set_logger(new_logger)
    
    env.env_method("set_mode", "adversary")
    adversary = SAC(**sac_kwargs)

    callback = RobustnessCallback(check_freq=1000)

    # Training Loop
    for r in range(1, ROUNDS + 1):
        budget = min(15.0 * r, 100.0)
        env.set_attr("max_budget", budget)
        print(f" > Round {r}/{ROUNDS} (Budget: {budget})")
        
        # 1. Train Protagonist
        adversary.save("tmp_adv")
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", "tmp_adv.zip", "SAC") 
        
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, 
                          tb_log_name="SAC_Protagonist", callback=callback)
        protagonist.save(f"{path_model}/protagonist/round_{r}")

        # 2. Train Adversary
        protagonist.save("tmp_prot")
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", "tmp_prot.zip", "SAC")
        
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="SAC_Adversary")
        adversary.save(f"{path_model}/adversary/round_{r}")

    env.close()
    if os.path.exists("tmp_adv.zip"): os.remove("tmp_adv.zip")
    if os.path.exists("tmp_prot.zip"): os.remove("tmp_prot.zip")

if __name__ == "__main__":
    for s in SCENARIOS:
        run_scenario(s)