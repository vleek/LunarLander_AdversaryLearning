import os
# --- OPTIMIZATION FLAGS ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import multiprocessing
import shutil
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- GLOBAL CONFIG ---
ROUNDS = 10
STEPS_PER_ROUND = 500_000 
MAX_WIND_FORCE = 15.0
NUM_CORES = 64 

# Base Directories
BASE_MODELS_DIR = "models_parallel"
BASE_LOGS_DIR = "logs_parallel"

# --- THE 6 SCENARIOS ---
SCENARIOS = [
    {"name": "PPO_Visible",  "algo": "PPO",  "visible": True},
    {"name": "PPO_Latent",   "algo": "PPO",  "visible": False},
    
    {"name": "LSTM_Visible", "algo": "LSTM", "visible": True},
    {"name": "LSTM_Latent",  "algo": "LSTM", "visible": False},
    
    {"name": "SAC_Visible",  "algo": "SAC",  "visible": True},
    {"name": "SAC_Latent",   "algo": "SAC",  "visible": False},
]

def make_env(rank: int, visible_wind: bool, log_dir: str, seed: int = 0):
    def _init():
        env = gym.make("LunarLander-v3", render_mode=None)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        env = AdversarialLanderWrapper(
            env, 
            max_wind_force=MAX_WIND_FORCE, 
            visible_wind=visible_wind
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def get_hyperparameters(algo_type, env):
    # 1. PPO CONFIG
    ppo_kwargs = {
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
    
    # 2. LSTM CONFIG
    if algo_type == "LSTM":
        lstm_kwargs = ppo_kwargs.copy()
        lstm_kwargs["policy"] = "MlpLstmPolicy"
        lstm_kwargs["policy_kwargs"] = {"enable_critic_lstm": False}
        return RecurrentPPO, lstm_kwargs

    # 3. SAC CONFIG
    elif algo_type == "SAC":
        sac_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "device": "cpu",
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "ent_coef": "auto",
            "gamma": 0.995,
            "train_freq": (1, "step"),
            "gradient_steps": 32,
        }
        return SAC, sac_kwargs
        
    return PPO, ppo_kwargs

def run_scenario(scenario):
    name = scenario["name"]
    algo_type = scenario["algo"]
    is_visible = scenario["visible"]
    
    print(f"\n{'#'*40}")
    print(f"STARTING SCENARIO: {name}")
    print(f"Algorithm: {algo_type} | Visible Wind: {is_visible}")
    print(f"{'#'*40}\n")

    # --- NEW DIRECTORY STRUCTURE ---
    # Structure: models_parallel/{Scenario_Name}/{Protagonist|Adversary}/
    scenario_model_dir = os.path.join(BASE_MODELS_DIR, name)
    prot_model_dir = os.path.join(scenario_model_dir, "protagonist")
    adv_model_dir = os.path.join(scenario_model_dir, "adversary")
    
    # Logs: logs_parallel/{Scenario_Name}/
    scenario_log_dir = os.path.join(BASE_LOGS_DIR, name)

    os.makedirs(prot_model_dir, exist_ok=True)
    os.makedirs(adv_model_dir, exist_ok=True)
    os.makedirs(scenario_log_dir, exist_ok=True)

    # 1. Create Env
    env = SubprocVecEnv([make_env(i, is_visible, scenario_log_dir) for i in range(NUM_CORES)])

    # 2. Setup Agents
    AgentClass, kwargs = get_hyperparameters(algo_type, env)
    kwargs["tensorboard_log"] = scenario_log_dir
    
    env.env_method("set_mode", "protagonist")
    protagonist = AgentClass(**kwargs)
    
    env.env_method("set_mode", "adversary")
    adversary = AgentClass(**kwargs)

    # 3. Training Loop
    for round_idx in range(1, ROUNDS + 1):
        print(f" > Round {round_idx}/{ROUNDS} [{name}]")
        
        current_budget = min(20.0 * round_idx, 100.0)
        env.set_attr("max_budget", current_budget)
        
        # --- TRAIN PROTAGONIST ---
        adv_path = "temp_adversary_worker"
        adversary.save(adv_path)
        
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", f"{adv_path}.zip", algo_type) 
        
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo_type}_Protagonist")
        # Save to specific subfolder
        protagonist.save(os.path.join(prot_model_dir, f"round_{round_idx}"))

        # --- TRAIN ADVERSARY ---
        prot_path = "temp_protagonist_worker"
        protagonist.save(prot_path)
        
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", f"{prot_path}.zip", algo_type)
        
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo_type}_Adversary")
        # Save to specific subfolder
        adversary.save(os.path.join(adv_model_dir, f"round_{round_idx}"))

    env.close()
    
    if os.path.exists(f"{adv_path}.zip"): os.remove(f"{adv_path}.zip")
    if os.path.exists(f"{prot_path}.zip"): os.remove(f"{prot_path}.zip")
    
    print(f"Scenario {name} Complete.")

def main():
    print(f"Detected {multiprocessing.cpu_count()} CPUs.")
    print(f"Using {NUM_CORES} cores.")
    
    for scenario in SCENARIOS:
        run_scenario(scenario)

if __name__ == "__main__":
    main()