import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class AdversarialLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_wind_force=5.0, max_budget=100.0, visible_wind=False):
        super().__init__(env)
        self.env = env
        
        # Config
        self.max_wind_force = max_wind_force
        self.max_budget = max_budget
        self.current_budget = max_budget
        self.visible_wind = visible_wind
        
        # Rate Limit (Prevents wind flickering)
        self.max_wind_change = 1.0 
        
        # --- NEW: SAFETY WARM-UP ---
        # The agent needs ~200k steps to learn basic flying.
        # We scale the effective wind force from 0% to 100% over this period.
        self.total_steps = 0
        self.warmup_period = 200_000 
        # ---------------------------

        # State
        self.opponent_model = None 
        self.mode = "protagonist" 
        self.current_wind = np.zeros(2, dtype=np.float32)

        # Spaces
        self.lander_action_space = env.action_space
        self.wind_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Obs Space Update
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
        assert mode in ["protagonist", "adversary"], "Invalid mode"
        self.mode = mode
        self.action_space = self.lander_action_space if mode == "protagonist" else self.wind_action_space

    def set_opponent(self, model_or_path, algo_type="PPO"):
        if isinstance(model_or_path, str):
            from stable_baselines3 import PPO, SAC
            from sb3_contrib import RecurrentPPO
            
            loader_map = {"PPO": PPO, "SAC": SAC, "LSTM": RecurrentPPO}
            loader = loader_map.get(algo_type, PPO)
            self.opponent_model = loader.load(model_or_path, device="cpu")
        else:
            self.opponent_model = model_or_path

    def _get_obs(self, obs):
        norm_budget = np.array([self.current_budget / self.max_budget], dtype=np.float32)
        enhanced_obs = np.concatenate([obs, norm_budget])
        
        if self.visible_wind:
            enhanced_obs = np.concatenate([enhanced_obs, self.current_wind])
            
        return enhanced_obs.astype(np.float32)

    def reset(self, **kwargs):
        self.current_budget = self.max_budget
        self.current_wind = np.zeros(2, dtype=np.float32)
        
        obs, info = self.env.reset(**kwargs)
        self.last_obs = self._get_obs(obs)
        return self.last_obs, info

    def step(self, action):
        lander_action = None
        target_wind_raw = np.array([0.0, 0.0], dtype=np.float32)
        self.total_steps += 1 # Count steps for warm-up

        # 1. Resolve Actions
        if self.mode == "protagonist":
            lander_action = action
            if self.opponent_model and self.current_budget > 0:
                prediction, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
                target_wind_raw = prediction
                
        elif self.mode == "adversary":
            target_wind_raw = action
            if self.opponent_model:
                lander_action, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
            else:
                lander_action = 0 

        # Shape Check
        target_wind_raw = np.array(target_wind_raw, dtype=np.float32).flatten()
        if target_wind_raw.size < 2:
            target_wind_raw = np.resize(target_wind_raw, (2,)) 
            target_wind_raw[1] = 0.0
        target_wind_raw = target_wind_raw[:2]

        # 2. Wind Calculation
        target_wind_vector = target_wind_raw * self.max_wind_force
        
        # Rate Limiting (Smoothing)
        delta = target_wind_vector - self.current_wind
        delta = np.clip(delta, -self.max_wind_change, self.max_wind_change)
        wind_vector = self.current_wind + delta

        # --- SAFETY WARM-UP SCALING ---
        # If total steps < 200k, multiply wind by a factor (0.0 to 1.0)
        # This guarantees the agent isn't crushed in the first hour of training.
        if self.total_steps < self.warmup_period:
            warmup_factor = self.total_steps / self.warmup_period
            wind_vector *= warmup_factor
        # ------------------------------

        # Budget Check
        force_magnitude = np.linalg.norm(wind_vector)
        cost = (force_magnitude * 0.05) + (force_magnitude**2 * 0.005)
        
        self.current_budget -= cost
        if self.current_budget <= 0:
            self.current_budget = 0.0
            wind_vector = np.zeros(2, dtype=np.float32)

        self.current_wind = wind_vector

        # 3. Apply Physics
        try:
            lander = self.env.unwrapped.lander
            if lander:
                lander.ApplyForceToCenter((float(wind_vector[0]), float(wind_vector[1])), True)
        except AttributeError:
            pass

        # 4. Step
        next_obs_raw, reward, terminated, truncated, info = self.env.step(lander_action)
        
        # Adversary Rewards (Sniper Logic)
        pos_y = next_obs_raw[1]
        angle = next_obs_raw[4]
        is_vulnerable = (pos_y < 0.5) or (abs(angle) > 0.2)
        sniper_bonus = 0.0
        
        if is_vulnerable and force_magnitude > 1.0:
            sniper_bonus = 0.05 

        next_obs = self._get_obs(next_obs_raw)
        self.last_obs = next_obs
        
        if self.mode == "adversary":
            reward = -reward 
            reward -= (cost * 0.5) 
            reward += sniper_bonus

        info["budget"] = self.current_budget
        return next_obs, reward, terminated, truncated, info