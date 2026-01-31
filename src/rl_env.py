import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartGridEnv(gym.Env):
    def __init__(self, prices, base_load):
        super(SmartGridEnv, self).__init__()

        self.prices = prices
        self.base_load = base_load
        self.current_step = 0

        # Action: 0 = delay appliance, 1 = run appliance
        self.action_space = spaces.Discrete(2)

        # Observation: [hour, price, base_load]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([23, 100, 1000], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        hour = self.current_step % 24
        return np.array([hour, self.prices[self.current_step], self.base_load[self.current_step]], dtype=np.float32)

    def step(self, action):
        price = self.prices[self.current_step]
        load = self.base_load[self.current_step]

        appliance_kwh = 2.0  # assume fixed appliance usage
        total_load = load + (appliance_kwh if action == 1 else 0)

        # cost
        cost = total_load * price

        # peak penalty (discourage peak time usage)
        peak_penalty = 5 if total_load > np.percentile(self.base_load, 90) else 0

        reward = -(cost + peak_penalty)

        self.current_step += 1
        done = self.current_step >= len(self.base_load) - 1

        return self._get_obs(), reward, done, False, {}
