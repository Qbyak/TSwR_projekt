import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racing_env import RacingEnv
import gymnasium as gym

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

def train_ai():

    env_kwargs = {
        'csv_path': 'BrandsHatch.csv',
        'otl_path': None,
        'scale': 0.4
    }

    # Mój procesor ma 12 wątków, zostawiam sobie zapas
    n_envs = 10

    env = make_vec_env(
        RacingEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    model_path = "ppo_model.zip"

    if os.path.exists(model_path):
        model = PPO.load("ppo_model", env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=832, batch_size=64)
    model.learn(total_timesteps=200000)

    model.save("ppo_model")
    print("INFO: Model zaktualizowany i zapisany poprawnie.")


if __name__ == "__main__":
    train_ai()