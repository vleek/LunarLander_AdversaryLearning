import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment.adversarial_wrapper import AdversarialLanderWrapper

# Configuration 
ROUNDS = 10              
STEPS_PER_ROUND = 50000
MODELS_DIR = "models"
LOG_DIR = "logs"

# Experiment Settings
VISIBLE_WIND = False    # False = Latent (Hidden) Wind; True = Observable Wind
MAX_WIND_FORCE = 10.0

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    # 1. Environment Setup
    env = gym.make("LunarLander-v3", render_mode=None)
    env = Monitor(env, LOG_DIR) 
    env = AdversarialLanderWrapper(
        env, 
        max_wind_force=MAX_WIND_FORCE, 
        visible_wind=VISIBLE_WIND
    )

    # 2. Agent Initialization
    print(f"Initializing Agents (Latent Wind: {not VISIBLE_WIND})...")
    
    # Initialize Protagonist
    env.set_mode("protagonist")
    protagonist = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    
    # Initialize Adversary
    # Toggle mode briefly to ensure PPO infers the correct action space (2D Wind Vector)
    env.set_mode("adversary")
    adversary = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    # 3. Iterative Training Loop (RARL)
    for round_idx in range(1, ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx} / {ROUNDS} {'='*20}")

        # Curriculum Learning: Ramp up adversary budget to prevent early-game spamming
        current_budget = min(20.0 * round_idx, 100.0)
        env.max_budget = current_budget
        print(f"Adversary Budget set to: {current_budget:.1f}")

        # Phase A: Train Protagonist
        print(f"> Training Protagonist (vs Fixed Adversary)...")
        env.set_mode("protagonist")
        env.set_opponent(adversary)
        
        protagonist.learn(
            total_timesteps=STEPS_PER_ROUND, 
            reset_num_timesteps=False, 
            tb_log_name="PPO_Protagonist"
        )
        
        prot_path = f"{MODELS_DIR}/protagonist_round_{round_idx}"
        protagonist.save(prot_path)

        # Phase B: Train Adversary
        print(f"> Training Adversary (vs Fixed Protagonist)...")
        env.set_mode("adversary")
        env.set_opponent(protagonist)
        
        adversary.learn(
            total_timesteps=STEPS_PER_ROUND, 
            reset_num_timesteps=False, 
            tb_log_name="PPO_Adversary"
        )
        
        adv_path = f"{MODELS_DIR}/adversary_round_{round_idx}"
        adversary.save(adv_path)

    print("\nTraining Complete.")
    env.close()

if __name__ == "__main__":
    main()