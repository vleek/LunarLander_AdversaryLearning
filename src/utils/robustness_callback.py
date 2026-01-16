import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RobustnessMetricsCallback(BaseCallback):
    """
    Logs custom safety metrics to TensorBoard/CSV:
    1. CVaR (5%): The average reward of the worst 5% of episodes.
    2. Impact Velocity: How hard the lander hits the ground (crash intensity).
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Access recent episode data (requires Monitor wrapper)
            buf = self.model.ep_info_buffer
            
            if len(buf) > 0:
                rewards = [info["r"] for info in buf]
                # 'impact_vel' is a custom key added by our environment wrapper
                impacts = [info["impact_vel"] for info in buf if "impact_vel" in info]

                # 1. Calculate CVaR (Worst 5% Performance)
                # We need at least 20 episodes to statistically define "5%"
                if len(rewards) >= 20:
                    cutoff = max(1, int(len(rewards) * 0.05))
                    worst_episodes = np.sort(rewards)[:cutoff]
                    self.logger.record("robustness/cvar_5_percent", np.mean(worst_episodes))

                # 2. Calculate Average Crash Intensity
                if impacts:
                    self.logger.record("robustness/avg_impact_vel", np.mean(impacts))

        return True