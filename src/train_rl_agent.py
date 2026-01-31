import numpy as np
from stable_baselines3 import PPO
from src.rl_env import SmartGridEnv

def train_rl():
    # dummy example data
    prices = np.random.uniform(3, 10, 500)  # electricity price per unit
    base_load = np.random.uniform(10, 60, 500)  # base load

    env = SmartGridEnv(prices, base_load)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    model.save("models/ppo_scheduler")
    print("Saved PPO agent at models/ppo_scheduler.zip")

if __name__ == "__main__":
    train_rl()
