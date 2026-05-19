import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from racing_env import RacingEnv


def watch_agent():
    env = RacingEnv(csv_path='BrandsHatch.csv', scale=0.4,otl_path='BrandsHatch_otl.csv')

    model = PPO.load("ppo_model")
    obs, _ = env.reset()
    terminated = False
    truncated = False

    path_x = []
    path_y = []

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        state = env.car.get_state()
        car_x, car_y = env.track.get_global_coords(state['s'], state['n'])
        path_x.append(car_x)
        path_y.append(car_y)

    print(f"INFO: Jazda zakończona po {len(path_x)} krokach.")
    plt.figure(figsize=(10, 6))
    track_coords = np.array(env.track.polygon.exterior.coords)
    plt.plot(track_coords[:, 0], track_coords[:, 1], 'b-', label='Granice toru', alpha=0.5)


    otl_coords = np.array(env.track.center_line.coords)
    plt.plot(otl_coords[:, 0], otl_coords[:, 1], 'k--', label='Linia optymalna (OTL)', alpha=0.5)


    plt.plot(path_x, path_y, 'r-', linewidth=2, label='Trajektoria AI')
    plt.plot(path_x[0], path_y[0], 'go', label='Start')
    plt.plot(path_x[-1], path_y[-1], 'rx', markersize=10, label='Miejsce wypadku')

    plt.title("Analiza przejazdu")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    watch_agent()