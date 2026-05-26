import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from racing_env_Ld_const import RacingEnv
import gymnasium as gym

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = {}
        self.explained_variances = []
        self.ev_timesteps = []

    def _on_step(self):
        for i, done in enumerate(self.locals["dones"]):
            reward = self.locals["rewards"][i]
            self.current_rewards[i] = self.current_rewards.get(i, 0) + reward
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0
        return True

    def _on_rollout_end(self):
        ev = self.logger.name_to_value.get("train/explained_variance", None)
        if ev is not None:
            self.explained_variances.append(ev)
            self.ev_timesteps.append(self.num_timesteps)

def train_ai():

    env_kwargs = {
        'csv_path': 'Catalunya.csv',
        'otl_path': None,
        'scale': 0.4
    }

    n_envs = 10

    env = make_vec_env(
        RacingEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    model_path = "ppo_model_2505_Ld_2.zip"
    custom_parameters = {
        "learning_rate": 0.0001,
    }

    if os.path.exists(model_path):
        model = PPO.load("ppo_model_2505_Ld_2", env=env,custom_objects=custom_parameters)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=128
        )
    callback = TrainingLoggerCallback()
    model.learn(total_timesteps=2000000, callback=callback,reset_num_timesteps=False)
    """
    existing_reward= np.load("reward_log_2505_Ld.npy").tolist() if os.path.exists("reward_log_2505_6.npy") else []
    existing_ev = np.load("ev_log_2505_6.npy").tolist() if os.path.exists("ev_log_2505_6.npy") else []
    existing_ev_timestamps = np.load("ev_timesteps_2505_6.npy").tolist() if os.path.exists("ev_timesteps_2505_6.npy") else []
    all_rewards = existing_reward + callback.episode_rewards
    all_ev = existing_ev + callback.explained_variances
    all_timesteps = existing_ev_timestamps + callback.ev_timesteps


    np.save("reward_log_2505_6.npy", all_rewards)
    np.save("ev_log_2505_6.npy", all_ev)
    np.save("ev_timesteps_2505_6.npy", all_timesteps)
    """
    model.save("ppo_model_2505_Ld_2")
    print("INFO: Model   i zapisany poprawnie.")


if __name__ == "__main__":
    train_ai()