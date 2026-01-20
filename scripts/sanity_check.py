import os
import sys
import gymnasium as gym
import numpy as np

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Force single-thread for simplicity
os.environ["OMP_NUM_THREADS"] = "1"

from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIG ---
# Short duration: 3000 steps is enough to see if they start using wind
TEST_STEPS = 3000  
NUM_CORES = 2      # Keep it light

MODELS_TO_TEST = [
    {"algo": "PPO",  "continuous": False}, # PPO usually trains on Discrete Lander
    {"algo": "LSTM", "continuous": False}, # LSTM usually trains on Discrete Lander
    {"algo": "SAC",  "continuous": True},  # SAC MUST have Continuous Lander
]

def make_env(rank, continuous_lander):
    def _init():
        # 1. Create Base Env (Switch Continuous/Discrete based on algo)
        env = gym.make("LunarLander-v3", continuous=continuous_lander, render_mode=None)
        
        # 2. Add Adversarial Wrapper (Wind is ALWAYS continuous)
        # We enforce strict budget here to see if they use it
        env = AdversarialLanderWrapper(env, max_wind_force=10.0, max_budget=100.0, visible_wind=True)
        
        # 3. Monitor (Critical: To read 'budget' from info)
        env = Monitor(env) 
        env.reset(seed=rank)
        return env
    return _init

def run_comprehensive_check():
    print("--- COMPREHENSIVE ADVERSARY CHECK ---")
    print(f"Goal: Verify PPO, LSTM, and SAC all spend budget with new Cost (0.02) & Bonus (0.10)\n")
    
    for cfg in MODELS_TO_TEST:
        algo = cfg["algo"]
        cont = cfg["continuous"]
        print(f"> Testing {algo} (Lander Continuous={cont})...")
        
        # 1. Setup Env
        # Use DummyVecEnv for LSTM (simpler debugging) or Subproc for speed
        if algo == "LSTM":
            env = DummyVecEnv([make_env(0, cont)])
        else:
            env = SubprocVecEnv([make_env(i, cont) for i in range(NUM_CORES)])
        
        # 2. Setup Model
        try:
            if algo == "PPO":
                model = PPO("MlpPolicy", env, device="cpu", verbose=0)
            elif algo == "LSTM":
                model = RecurrentPPO("MlpLstmPolicy", env, device="cpu", verbose=0)
            elif algo == "SAC":
                model = SAC("MlpPolicy", env, device="cpu", verbose=0)
            
            # 3. Train Adversary (Short Burst)
            env.env_method("set_mode", "adversary")
            env.env_method("set_opponent", None) # No opponent, just wind practice
            
            model.learn(total_timesteps=TEST_STEPS)
            
            # 4. Check Vital Signs
            # Get budget from the environment wrapper
            # We access the internal wrapper attribute 'current_budget'
            budgets = env.env_method("get_wrapper_attr", "current_budget")
            avg_budget_left = np.mean(budgets)
            used_budget = 100.0 - avg_budget_left
            
            status = "✅ PASS" if used_budget > 1.0 else "❌ FAIL (Lazy)"
            if used_budget > 95.0: status = "⚠️ PASS (Manic)"
            
            print(f"  Result: {status} | Budget Used: {used_budget:.1f}%")
            
        except Exception as e:
            print(f"  ❌ CRASH: {e}")
        
        finally:
            env.close()
        print("-" * 40)

if __name__ == "__main__":
    run_comprehensive_check()