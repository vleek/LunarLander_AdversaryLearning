import gymnasium as gym
import time
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- Configuration ---
PROTAGONIST_PATH = "models/protagonist_round_10"
ADVERSARY_PATH   = "models/adversary_round_10"
NO_ADVERSARY     = False
PLOT_WIND        = True   # Set False to disable the live graph
MAX_WIND         = 5.0

def run_battle():
    # 1. Environment Setup
    print("Initializing Arena...")
    env = gym.make("LunarLander-v3", render_mode="human")
    env = AdversarialLanderWrapper(env, max_wind_force=MAX_WIND)

    # 2. Load Models
    print(f"Loading Protagonist: {PROTAGONIST_PATH}")
    pilot = PPO.load(PROTAGONIST_PATH)
    
    if not NO_ADVERSARY:
        print(f"Loading Adversary: {ADVERSARY_PATH}")
        wind_god = PPO.load(ADVERSARY_PATH)
        env.set_mode("protagonist")
        env.set_opponent(wind_god)
    else:
        print("Adversary Disabled (Calm Weather).")
        env.set_mode("protagonist")
        env.set_opponent(None)

    # 3. Plotting Setup
    line_x, line_y, fig = None, None, None
    wind_x_data = [0] * 100
    wind_y_data = [0] * 100

    if PLOT_WIND:
        print("Initializing Wind Monitor...")
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        line_x, = ax.plot(wind_x_data, color='red', label='Wind X')
        line_y, = ax.plot(wind_y_data, color='blue', label='Wind Y')
        
        ax.set_ylim(-MAX_WIND - 1, MAX_WIND + 1)
        ax.set_xlim(0, 100)
        ax.set_title("Live Wind Force")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 4. Battle Loop
    episodes = 5
    print(f"\nStarting {episodes} episodes...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Reset plot buffers
        wind_x_data = [0] * 100
        wind_y_data = [0] * 100
        
        print(f"\n--- Episode {ep + 1} ---")
        
        while not done:
            action, _ = pilot.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Update Stats
            budget = info.get('budget', 0.0)
            current_wind = getattr(env, "current_wind", [0.0, 0.0])
            
            # Update Plot
            if PLOT_WIND:
                wind_x_data.pop(0)
                wind_y_data.pop(0)
                wind_x_data.append(current_wind[0])
                wind_y_data.append(current_wind[1])
                
                line_x.set_ydata(wind_x_data)
                line_y.set_ydata(wind_y_data)
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Terminal HUD
            sys.stdout.write(f"\rScore: {reward:6.2f} | Budget: {budget:6.1f} | Wind X: {current_wind[0]:5.2f}")
            sys.stdout.flush()
            
            time.sleep(0.01) 

        status = "CRASHED" if total_reward < 100 else "LANDED"
        print(f"\nEpisode {ep + 1} Finished. Score: {total_reward:.2f} [{status}]")

    if PLOT_WIND:
        plt.ioff()
        plt.show()
    env.close()

if __name__ == "__main__":
    run_battle()