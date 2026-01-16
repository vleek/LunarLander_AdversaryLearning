import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AdversarialLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_wind_force=5.0, max_budget=100.0, visible_wind=False):
        super().__init__(env)
        self.env = env
        
        self.max_wind_force = max_wind_force
        self.max_budget = max_budget
        self.current_budget = max_budget
        self.visible_wind = visible_wind
        self.max_wind_change = 1.0 
        
        # Warm-up: Scale physical wind from 0% to 100% over first 200k steps.
        # FIX: We now decouple observation from physics so the adversary learns immediately.
        self.total_steps = 0
        self.warmup_period = 200_000 

        self.opponent_model = None 
        self.mode = "protagonist" 
        self.current_wind = np.zeros(2, dtype=np.float32)

        # Define Spaces
        self.lander_action_space = env.action_space
        self.wind_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Update Observation Space (Add Budget + Optional Wind)
        env_low = self.env.observation_space.low.astype(np.float32)
        env_high = self.env.observation_space.high.astype(np.float32)
        extra_dims = 1 + (2 if visible_wind else 0) 
        
        low = np.concatenate([env_low, [0.0] * extra_dims]).astype(np.float32)
        high = np.concatenate([env_high, [1.0] * extra_dims]).astype(np.float32)
        
        if visible_wind:
            high[-2:] = float(max_wind_force) 
            low[-2:] = float(-max_wind_force)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_mode(self, mode):
        self.mode = mode
        self.action_space = self.lander_action_space if mode == "protagonist" else self.wind_action_space

    def set_opponent(self, model_or_path, algo_type="PPO"):
        if isinstance(model_or_path, str):
            from stable_baselines3 import PPO, SAC
            from sb3_contrib import RecurrentPPO
            loader = {"PPO": PPO, "SAC": SAC, "LSTM": RecurrentPPO}.get(algo_type, PPO)
            self.opponent_model = loader.load(model_or_path, device="cpu")
        else:
            self.opponent_model = model_or_path

    def _get_obs(self, obs):
        norm_budget = np.array([self.current_budget / self.max_budget], dtype=np.float32)
        enhanced = np.concatenate([obs, norm_budget])
        if self.visible_wind:
            enhanced = np.concatenate([enhanced, self.current_wind])
        return enhanced.astype(np.float32)

    def reset(self, **kwargs):
        self.current_budget = self.max_budget
        self.current_wind = np.zeros(2, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        self.last_obs = self._get_obs(obs)
        return self.last_obs, info

    def step(self, action):
        self.total_steps += 1
        lander_action, target_wind = None, np.zeros(2, dtype=np.float32)

        # 1. Determine Actions based on Role
        if self.mode == "protagonist":
            lander_action = action
            if self.opponent_model and self.current_budget > 0:
                target_wind, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
        else:
            target_wind = action
            if self.opponent_model:
                lander_action, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
            else:
                lander_action = 0 

        # 2. Process Wind (Target -> Internal State)
        target_wind = np.resize(np.array(target_wind).flatten(), (2,))[:2]
        target_vec = target_wind * self.max_wind_force
        
        # Smooth changes (Updates self.current_wind - what the agent SEES)
        delta = np.clip(target_vec - self.current_wind, -self.max_wind_change, self.max_wind_change)
        self.current_wind = self.current_wind + delta

        # --- FIX: Decouple Observation from Physics ---
        # The agent sees and pays for 'self.current_wind' (Full intent).
        # But the physics engine applies 'applied_wind' (Dampened).
        applied_wind = self.current_wind.copy()
        
        if self.total_steps < self.warmup_period:
            warmup_factor = self.total_steps / self.warmup_period
            # Ensure at least 10% force flows through so causality is learned
            applied_wind *= max(0.1, warmup_factor)
        # ----------------------------------------------

        # Apply Budget Costs (Based on INTENT, not dampening)
        force = np.linalg.norm(self.current_wind)
        cost = (force * 0.05) + (force**2 * 0.005)
        self.current_budget -= cost
        if self.current_budget <= 0:
            self.current_wind = np.zeros(2)
            applied_wind = np.zeros(2)
            self.current_budget = 0
            
        # 3. Apply Physics (Use Dampened Wind)
        try:
            self.env.unwrapped.lander.ApplyForceToCenter((float(applied_wind[0]), float(applied_wind[1])), True)
        except AttributeError:
            pass

        # 4. Step Environment
        next_obs_raw, reward, terminated, truncated, info = self.env.step(lander_action)

        # Log Impact Velocity
        if terminated or truncated:
            info["impact_vel"] = abs(next_obs_raw[3]) 

        # 5. Calculate Rewards
        if self.mode == "adversary":
            sniper_bonus = 0.05 if (next_obs_raw[1] < 0.5 or abs(next_obs_raw[4]) > 0.2) and force > 1.0 else 0.0
            reward = -reward - (cost * 0.5) + sniper_bonus

        next_obs = self._get_obs(next_obs_raw)
        self.last_obs = next_obs
        info["budget"] = self.current_budget
        
        return next_obs, reward, terminated, truncated, info