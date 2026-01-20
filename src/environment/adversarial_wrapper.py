import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AdversarialLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_wind_force=10.0, max_budget=100.0, visible_wind=False, adversary_noise=0.1):
        super().__init__(env)
        self.env = env
        
        self.max_wind_force = max_wind_force
        self.max_budget = max_budget
        self.current_budget = max_budget
        self.visible_wind = visible_wind
        self.max_wind_change = 1.0 
        
        # New: Noise Level (Standard Deviation of Gaussian Noise)
        self.adversary_noise = adversary_noise

        # Warm-up settings
        self.total_steps = 0
        self.warmup_period = 200_000 

        self.opponent_model = None 
        self.mode = "protagonist" 
        self.current_wind = np.zeros(2, dtype=np.float32)

        # Spaces
        self.lander_action_space = env.action_space
        self.wind_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space
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

        # 1. Action Resolution
        if self.mode == "protagonist":
            lander_action = action
            if self.opponent_model and self.current_budget > 0:
                target_wind, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
        else:
            target_wind = action
            if self.opponent_model:
                lander_action, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
            else:
                # Handle No-Op
                if isinstance(self.lander_action_space, spaces.Box):
                    lander_action = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    lander_action = 0
        
        # --- NEW: Inject Noise into Adversary Action ---
        # This makes the wind slightly unpredictable, improving robustness.
        if self.adversary_noise > 0.0:
            noise = np.random.normal(0, self.adversary_noise, size=np.array(target_wind).shape)
            target_wind = np.clip(target_wind + noise, -1.0, 1.0)

        # 2. Wind & Cost
        # Hard limit: Target vector cannot exceed max_wind_force
        target_vec = np.resize(np.array(target_wind).flatten(), (2,))[:2] * self.max_wind_force
        
        # Smooth transition (Physics limit)
        delta = np.clip(target_vec - self.current_wind, -self.max_wind_change, self.max_wind_change)
        self.current_wind = self.current_wind + delta

        # Warm-up (Decouple physics from observation early on)
        applied_wind = self.current_wind.copy()
        if self.total_steps < self.warmup_period:
            warmup_factor = self.total_steps / self.warmup_period
            applied_wind *= max(0.1, warmup_factor)

        # Cost Calculation (Cheaper: 0.02)
        force = np.linalg.norm(self.current_wind)
        cost = force * 0.02  
        
        self.current_budget -= cost
        if self.current_budget <= 0:
            self.current_budget = 0.0
            self.current_wind = np.zeros(2)
            applied_wind = np.zeros(2)
            
        # 3. Physics
        try:
            self.env.unwrapped.lander.ApplyForceToCenter((float(applied_wind[0]), float(applied_wind[1])), True)
        except AttributeError:
            pass

        # 4. Step
        next_obs_raw, reward, terminated, truncated, info = self.env.step(lander_action)

        if terminated or truncated:
            info["impact_vel"] = abs(next_obs_raw[3])

        # 5. REWARD LOGIC
        if self.mode == "adversary":
            adv_reward = -reward 
            adv_reward -= cost 

            # Incentive to be active (High Bonus: 0.10)
            if force > 2.0:
                adv_reward += 0.10 

            is_low = (next_obs_raw[1] < 0.2)
            is_unstable = (abs(next_obs_raw[4]) > 0.15)
            
            is_vulnerable = is_low or is_unstable

            if is_vulnerable and force > 5.0:
                 adv_reward += 0.1 

            reward = adv_reward

        next_obs = self._get_obs(next_obs_raw)
        self.last_obs = next_obs
        info["budget"] = self.current_budget
        
        return next_obs, reward, terminated, truncated, info