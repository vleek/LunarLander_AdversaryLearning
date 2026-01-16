import os
import sys
import multiprocessing
import shutil
import gymnasium as gym

# --- 1. PATH SETUP (Critical for importing from 'src') ---
# Add the project root directory to the python path so we can see 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Force single-thread per core to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Config ---
ROUNDS = 10
STEPS_PER_ROUND = 500_000 
MAX_WIND_FORCE = 15.0
NUM_CORES = 64  

# Paths updated to point to the new folder structure
# They go up one level ("..") then into "checkpoints" or "logs"
BASE_MODELS = os.path.join("..", "checkpoints", "models_parallel")
BASE_LOGS = os.path.join("..", "logs_parallel")

SCENARIOS = [
    {"name": "PPO_Visible",  "algo": "PPO",  "visible": True},
    {"name": "PPO_Latent",   "algo": "PPO",  "visible": False},
    {"name": "LSTM_Visible", "algo": "LSTM", "visible": True},
    {"name": "LSTM_Latent",  "algo": "LSTM", "visible": False},
]

def make_env(rank, visible_wind, log_dir, seed=0):
    def _init():
        env = gym.make("LunarLander-v3", render_mode=None)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND_FORCE, visible_wind=visible_wind)
        env.reset(seed=seed + rank)
        return env
    return _init

def get_model_config(algo_type, env):
    # Standard PPO settings optimized for CPU parallelization
    config = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "device": "cpu",
        "n_steps": 128,      
        "batch_size": 2048,
        "n_epochs": 5,
        "learning_rate": 3e-4, 
        "ent_coef": 0.01,
        "gamma": 0.995,
    }
    
    if algo_type == "LSTM":
        config["policy"] = "MlpLstmPolicy"
        config["policy_kwargs"] = {"enable_critic_lstm": False} # Optimization
        return RecurrentPPO, config

    return PPO, config

def run_scenario(scenario):
    name, algo, vis = scenario["name"], scenario["algo"], scenario["visible"]
    print(f"\n>>> Starting: {name} ({algo})")

    # Setup paths
    path_model = os.path.join(BASE_MODELS, name)
    path_log = os.path.join(BASE_LOGS, name)
    os.makedirs(f"{path_model}/protagonist", exist_ok=True)
    os.makedirs(f"{path_model}/adversary", exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # Init parallel envs
    env = SubprocVecEnv([make_env(i, vis, path_log) for i in range(NUM_CORES)])

    # Init agents
    ModelClass, kwargs = get_model_config(algo, env)
    kwargs["tensorboard_log"] = path_log
    
    env.env_method("set_mode", "protagonist")
    protagonist = ModelClass(**kwargs)
    
    env.env_method("set_mode", "adversary")
    adversary = ModelClass(**kwargs)

    # Training Loop
    for r in range(1, ROUNDS + 1):
        budget = min(20.0 * r, 100.0)
        env.set_attr("max_budget", budget)
        print(f" > Round {r}/{ROUNDS} (Budget: {budget})")
        
        # 1. Train Protagonist (Freeze Adversary)
        adversary.save("tmp_adv")
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", "tmp_adv.zip", algo) 
        
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo}_Protagonist")
        protagonist.save(f"{path_model}/protagonist/round_{r}")

        # 2. Train Adversary (Freeze Protagonist)
        protagonist.save("tmp_prot")
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", "tmp_prot.zip", algo)
        
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo}_Adversary")
        adversary.save(f"{path_model}/adversary/round_{r}")

    env.close()
    
    # Cleanup
    if os.path.exists("tmp_adv.zip"): os.remove("tmp_adv.zip")
    if os.path.exists("tmp_prot.zip"): os.remove("tmp_prot.zip")
    print(f"Finished {name}")

if __name__ == "__main__":
    print(f"Using {NUM_CORES} cores.")
    for s in SCENARIOS:
        run_scenario(s)