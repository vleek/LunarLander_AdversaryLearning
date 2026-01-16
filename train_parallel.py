import os
import multiprocessing
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Optimization Flags ---
# Forces numerical libraries to use a single thread to avoid CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Configuration ---
ROUNDS = 10
STEPS_PER_ROUND = 500_000 
MAX_WIND_FORCE = 15.0
NUM_CORES = 64  # Adjust based on your machine

BASE_MODELS_DIR = "models_parallel"
BASE_LOGS_DIR = "logs_parallel"

# --- Scenarios ---
SCENARIOS = [
    # Standard PPO Baselines
    # {"name": "PPO_Visible",  "algo": "PPO",  "visible": True,  "continuous": False},
    # {"name": "PPO_Latent",   "algo": "PPO",  "visible": False, "continuous": False},
    
    # LSTM Memory Agents
    # {"name": "LSTM_Visible", "algo": "LSTM", "visible": True,  "continuous": False},
    # {"name": "LSTM_Latent",  "algo": "LSTM", "visible": False, "continuous": False},
    
    # SAC Agents (Requires Continuous Environment)
    {"name": "SAC_Visible",  "algo": "SAC",  "visible": True,  "continuous": True},
    {"name": "SAC_Latent",   "algo": "SAC",  "visible": False, "continuous": True},
]

def make_env(rank, visible_wind, log_dir, continuous, seed=0):
    """
    Creates a distinct environment instance for each CPU core.
    """
    def _init():
        env = gym.make("LunarLander-v3", continuous=continuous, render_mode=None)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        
        # Wrapper handles 'Safety Warm-up' automatically now
        env = AdversarialLanderWrapper(
            env, 
            max_wind_force=MAX_WIND_FORCE, 
            visible_wind=visible_wind
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def get_hyperparameters(algo_type, env):
    """
    Returns algorithm class and specific hyperparameters.
    """
    # 1. PPO (Standard)
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
    
    # 2. LSTM (Recurrent)
    if algo_type == "LSTM":
        lstm_kwargs = ppo_kwargs.copy()
        lstm_kwargs["policy"] = "MlpLstmPolicy"
        lstm_kwargs["policy_kwargs"] = {"enable_critic_lstm": False}
        return RecurrentPPO, lstm_kwargs

    # 3. SAC (Off-Policy) - Optimized for CPU Throughput
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
            # BATCHED UPDATES: Fixes the 'Stop-and-Go' CPU bottleneck
            "train_freq": (50, "step"),
            "gradient_steps": 1600,
        }
        return SAC, sac_kwargs
        
    return PPO, ppo_kwargs

def run_scenario(scenario):
    name = scenario["name"]
    algo = scenario["algo"]
    
    print(f"\n--- Starting Scenario: {name} ---")

    # Directory Setup
    scenario_dir = os.path.join(BASE_MODELS_DIR, name)
    log_dir = os.path.join(BASE_LOGS_DIR, name)
    os.makedirs(os.path.join(scenario_dir, "protagonist"), exist_ok=True)
    os.makedirs(os.path.join(scenario_dir, "adversary"), exist_ok=True)

    # Core Management: SAC requires fewer cores to avoid update overhead
    active_cores = 16 if algo == "SAC" else NUM_CORES

    # Initialize Parallel Environments
    env = SubprocVecEnv([
        make_env(i, scenario["visible"], log_dir, scenario["continuous"]) 
        for i in range(active_cores)
    ])

    # Initialize Agents
    AgentClass, kwargs = get_hyperparameters(algo, env)
    kwargs["tensorboard_log"] = log_dir
    
    env.env_method("set_mode", "protagonist")
    protagonist = AgentClass(**kwargs)
    
    env.env_method("set_mode", "adversary")
    adversary = AgentClass(**kwargs)

    # --- Training Curriculum ---
    for r in range(1, ROUNDS + 1):
        # Budget Formula: Linear ramp up to 100.0
        # The wrapper's internal warm-up handles the safety for Round 1
        budget = min(15.0 * r, 100.0)
        env.set_attr("max_budget", budget)
        
        print(f" > Round {r}/{ROUNDS} | Budget: {budget:.1f}")
        
        # 1. Train Protagonist (Against Frozen Adversary)
        adversary.save("tmp_adv")
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", f"tmp_adv.zip", algo) 
        
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo}_Protagonist")
        protagonist.save(os.path.join(scenario_dir, "protagonist", f"round_{r}"))

        # 2. Train Adversary (Against Frozen Protagonist)
        protagonist.save("tmp_prot")
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", f"tmp_prot.zip", algo)
        
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name=f"{algo}_Adversary")
        adversary.save(os.path.join(scenario_dir, "adversary", f"round_{r}"))

    # Cleanup
    env.close()
    for tmp in ["tmp_adv.zip", "tmp_prot.zip"]:
        if os.path.exists(tmp): os.remove(tmp)
    
    print(f"Scenario {name} Complete.")

if __name__ == "__main__":
    print(f"System: {multiprocessing.cpu_count()} CPUs detected.")
    for s in SCENARIOS:
        run_scenario(s)