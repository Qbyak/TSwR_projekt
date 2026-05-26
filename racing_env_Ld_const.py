import gymnasium as gym
from gymnasium import spaces
import numpy as np

from track import create_track
from vehicle import Vehicle
from controller import PurePursuitController


def get_curvature(track, s, epsilon=5.0):
    p1 = track.center_line.interpolate((s - epsilon) % track.length)
    p2 = track.center_line.interpolate(s % track.length)
    p3 = track.center_line.interpolate((s + epsilon) % track.length)
    theta1 = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    theta2 = np.arctan2(p3.y - p2.y, p3.x - p2.x)
    d_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return d_theta / epsilon


class RacingEnv(gym.Env):
    """Niestandardowe środowisko RL dla bolidu Formuły Student."""

    def __init__(self, csv_path='BrandsHatch.csv', scale=0.4, otl_path=None):
        super(RacingEnv, self).__init__()

        self.track = create_track(csv_path, scale, otl_path)
        self.dt = 0.025

        # Akcja [0]: Kontrola pedałów (target_T) [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # [vx, vy, błąd_n, kappa_now, kappa_ahead, kappa2, last_T]
        high = np.array([17.0, 10.0, 5.0, 1.0, 1.5,1.0,0.8, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.car = None
        self.controller = None
        self.current_step = 0
        self.max_steps = 20000
        self.last_T = 0.0
        self.prev_T_for_penalty = 0.0  # Pamięć poprzedniego kroku do wyliczania kary

    def reset(self, seed=None, options=None):
        """Resetuje środowisko na starcie nowego epizodu (okrążenia)."""
        super().reset(seed=seed)

        random_s = np.random.uniform(0, self.track.length)
        self.car = Vehicle(s0=random_s, n0=0.0, mu0=0.0, vx0=0.0)

        self.controller = PurePursuitController(wheelbase=1.5, max_steering=0.9)
        self.controller.last_delta = 0.0
        self.current_step = 0
        self.last_T = 0.0
        self.prev_T_for_penalty = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """Wykonuje jeden krok symulacji na podstawie decyzji AI."""
        self.current_step += 1
        state = self.car.get_state()
        vx = state["vx"]

        k = 1.2
        ai_Ld = np.clip(4, 2.0, 25.0)
        #ai_Ld = 12.0 + action[0] * 10.0
        curvature_now = get_curvature(self.track, state['s'], epsilon=2.0)


        raw_delta, _, _ = self.controller.compute_steering(state, self.track, ai_Ld)
        delta = 0.6 * self.controller.last_delta + 0.4 * raw_delta
        self.controller.last_delta = delta

        # Kontrola pedałów
        target_T = action[0]

        old_s = state['s']
        self.last_T = target_T

        # Aktualizacja fizyki bolidu
        self.car.update_dynamic(target_delta=delta, target_T=target_T, curvature=curvature_now, dt=self.dt)
        new_state = self.car.get_state()

        # Obliczenie Nagrody i warunków zakończenia
        reward, terminated, truncated = self._compute_reward_and_done(old_s, new_state)

        # Aktualizacja pamięci pedału na sam koniec kroku
        self.prev_T_for_penalty = target_T

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Zwraca wektor obserwacji dla sieci neuronowej."""
        state = self.car.get_state()
        kappa_now = get_curvature(self.track, state['s'], epsilon=2.0)
        kappa_ahead = get_curvature(self.track, state['s'] + max(state['vx'], 1.5), epsilon=2.0)
        kappa_2 = get_curvature(self.track, state['s'] + max(state['vx'] * 0.6 , 1.0), epsilon=2.0)
        kappa_3 = get_curvature(self.track, state['s'] + max(state['vx'] * 0.3, 0.8), epsilon=2.0)
        return np.array([
            state['vx'],
            self.car.vy,
            state['n'],
            kappa_now,
            kappa_ahead,
            kappa_2,
            kappa_3,
            self.last_T
        ], dtype=np.float32)

    def _compute_reward_and_done(self, old_s, current_state):
        """System kar i nagród wyścigowych."""
        terminated = False
        truncated = False
        reward = 0.0

        car_x, car_y = self.track.get_global_coords(current_state['s'], current_state['n'])
        ds = current_state['s'] - old_s


        if ds < -self.track.length / 2:
            ds += self.track.length


        survival_bonus = 0.5
        reward += survival_bonus

        speed_reward = current_state['vx'] * 0.1
        reward += speed_reward

        reward += ds * 1.02

        reward -= (current_state['n'] ** 2) * 2
        reward -= (current_state['vy'] ** 2) * 0.05

        #target_T_diff = abs(self.last_T - self.prev_T_for_penalty)
        #reward -= 0.02 * target_T_diff

        if not self.track.is_inside(car_x, car_y) and abs(current_state['n']) > 0.85:
            reward = -300.0
            terminated = True

            return reward, terminated, truncated

        if self.current_step >= self.max_steps:
            truncated = True

        return reward, terminated, truncated