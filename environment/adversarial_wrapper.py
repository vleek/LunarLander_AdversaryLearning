import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class AdversarialLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_wind_force=5.0, max_budget=100.0, visible_wind=False):
        super().__init__(env)
        self.env = env
        
        # Configuration
        self.max_wind_force = max_wind_force
        self.max_budget = max_budget
        self.current_budget = max_budget
        self.visible_wind = visible_wind
        
        # ### NEW: Rate Limiting Config ###
        # Maximum amount the wind force can change per frame (e.g., 1.0 unit per step)
        # This prevents "flickering" or instant full-power dumps.
        self.max_wind_change = 1.0 
        
        # State
        self.opponent_model = None 
        self.mode = "protagonist" 
        self.current_wind = np.zeros(2, dtype=np.float32)

        # Action Spaces
        self.lander_action_space = env.action_space
        self.wind_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- OBSERVATION SPACE UPDATE ---
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
        """
        Loads the opponent model based on the algorithm type.
        algo_type: "PPO", "SAC", or "LSTM"
        """
        if isinstance(model_or_path, str):
            # Dynamic Import to avoid circular dependencies
            from stable_baselines3 import PPO, SAC
            from sb3_contrib import RecurrentPPO
            
            # Map string to Class
            loader_map = {
                "PPO": PPO,
                "SAC": SAC,
                "LSTM": RecurrentPPO
            }
            loader = loader_map.get(algo_type, PPO)
                
            # Load model on CPU
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

        # 1. RESOLVE ACTIONS
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

        # --- SAFETY BLOCK: FORCE SHAPE (2,) ---
        target_wind_raw = np.array(target_wind_raw, dtype=np.float32).flatten()
        if target_wind_raw.size < 2:
            target_wind_raw = np.resize(target_wind_raw, (2,)) 
            target_wind_raw[1] = 0.0
        target_wind_raw = target_wind_raw[:2]

        # 2. CALCULATE WIND (WITH SMOOTHING)
        # Calculate the Target Vector (Where the network WANTS to be)
        target_wind_vector = target_wind_raw * self.max_wind_force
        
        # ### NEW: Rate Limiting Logic ###
        # Calculate difference between current wind and target
        delta = target_wind_vector - self.current_wind
        
        # Clip the change so it can't jump too fast (Smoothing)
        delta = np.clip(delta, -self.max_wind_change, self.max_wind_change)
        
        # Apply the smooth change
        wind_vector = self.current_wind + delta
        # ------------------------------

        # Check Budget
        force_magnitude = np.linalg.norm(wind_vector)
        cost = (force_magnitude * 0.05) + (force_magnitude**2 * 0.005)
        
        self.current_budget -= cost
        if self.current_budget <= 0:
            self.current_budget = 0.0
            wind_vector = np.zeros(2, dtype=np.float32) # Instant cut-off if out of fuel

        self.current_wind = wind_vector

        # 3. APPLY PHYSICS
        try:
            lander = self.env.unwrapped.lander
            if lander:
                lander.ApplyForceToCenter((float(wind_vector[0]), float(wind_vector[1])), True)
        except AttributeError:
            pass

        # 4. STEP ENVIRONMENT
        next_obs_raw, reward, terminated, truncated, info = self.env.step(lander_action)
        
        # ### NEW: State-Based "Sniper" Logic ###
        # Extract State (LunarLander-v3 state indices: 0=X, 1=Y, 2=VelX, 3=VelY, 4=Angle, etc.)
        pos_y = next_obs_raw[1]
        angle = next_obs_raw[4]
        
        # Define Vulnerability: Low altitude (< 0.5) OR High Tilt (> 0.2 rad)
        is_vulnerable = (pos_y < 0.5) or (abs(angle) > 0.2)
        sniper_bonus = 0.0
        
        # If we are attacking significantly while the agent is vulnerable, get a bonus
        if is_vulnerable and force_magnitude > 1.0:
            sniper_bonus = 0.05 # Small "Good Job" reward for timing
        # ---------------------------------------

        next_obs = self._get_obs(next_obs_raw)
        self.last_obs = next_obs
        
        if self.mode == "adversary":
            reward = -reward  # Zero-Sum Base
            reward -= (cost * 0.5) # Usage Tax (Prevents spraying)
            reward += sniper_bonus # Add Sniper Bonus

        info["budget"] = self.current_budget
        return next_obs, reward, terminated, truncated, info