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

    def __init__(self, csv_path='BrandsHatch.csv', scale=0.4,otl_path=None):
        super(RacingEnv, self).__init__()

        self.track = create_track(csv_path, scale, otl_path)
        self.dt = 0.025

        # Akcja [0]: Modyfikator odległości patrzenia (Ld)
        # Akcja [1]: Modyfikator agresywności hamowania (limit ay)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # [vx, vy, błąd_n, kappa_now, kappa_ahead]
        high = np.array([30.0, 10.0, 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.car = None
        self.controller = None
        self.current_step = 0
        self.max_steps = 20000

    def reset(self, seed=None, options=None):
        """Resetuje środowisko na starcie nowego epizodu (okrążenia)."""
        super().reset(seed=seed)

        # do testów losowy start
        random_s = np.random.uniform(0, self.track.length)
        #self.car = Vehicle(s0=random_s, n0=0.0, mu0=0.0, vx0=5.0)

        # start z początku toru
        self.car = Vehicle(s0=0.0, n0=0.0, mu0=0.0, vx0=5.0)
        self.controller = PurePursuitController(wheelbase=1.5, max_steering=0.9)
        self.controller.last_delta = 0.0
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        """Wykonuje jeden krok symulacji na podstawie decyzji AI."""
        self.current_step += 1
        state = self.car.get_state()
        vx = state["vx"]

        #
        # action[0] [-1, 1] -> Promień widzenia w przód (od 2.0 do 8.0 m)
        k = 0.3 + action[0] * 0.2  # AI tunes the gain, not the raw distance
        ai_Ld = np.clip(k * max(vx, 3.0), 2.0, 15.0)

        # action[1] [-1, 1] -> kontrola prędkości (od 2.0 do 18.0 m/s)
        v_ref = 10.0 + action[1] * 8.0

        curvature_now = get_curvature(self.track, state['s'], epsilon=5.0)

        # Sterowanie Pure Pursuit
        raw_delta, _, _ = self.controller.compute_steering(state, self.track, ai_Ld)
        delta = 0.2 * self.controller.last_delta + 0.8 * raw_delta
        self.controller.last_delta = delta

        # Kontrola pedałów
        target_T = 0.1 * (v_ref - vx)
        target_T = np.clip(target_T, -1.0, 1.0)
        #print(target_T)
        # Aktualizacja stanu
        old_s = state['s']
        self.car.update_dynamic(target_delta=delta, target_T=target_T, curvature=curvature_now, dt=self.dt)
        new_state = self.car.get_state()

        # Obliczenie Nagrody
        reward, terminated, truncated = self._compute_reward_and_done(old_s, new_state)

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Zwraca wektor obserwacji dla sieci neuronowej."""
        state = self.car.get_state()
        # Do skrętu
        kappa_now = get_curvature(self.track, state['s'], epsilon=5.0)
        # Do hamowania
        kappa_ahead = get_curvature(self.track, state['s'] + max(state['vx'], 1.0), epsilon=5.0)

        return np.array([
            state['vx'],
            self.car.vy,
            state['n'],
            kappa_now,
            kappa_ahead
        ], dtype=np.float32)

    def _compute_reward_and_done(self, old_s, current_state):
        """System kar i nagród."""
        terminated = False
        truncated = False
        reward = 0.0

        car_x, car_y = self.track.get_global_coords(current_state['s'], current_state['n'])
        ds = current_state['s'] - old_s
        # Zabezpieczenie przed przeskokiem s na linii start/meta
        if ds < -self.track.length / 2:
            ds += self.track.length

        survival_bonus = 0.6
        reward += survival_bonus

        # Nagroda za dystans
        reward += ds * 0.3

        # Kara za zjechanie z linii jazdy
        reward -= abs(current_state['n']) * 0.05

        # Kara za wypadek
        if not self.track.is_inside(car_x, car_y) or abs(current_state['n']) > 3.0:
            reward = -1000.0
            terminated = True
            return reward, terminated, truncated
        """
        # Przejechanie całego okrążenia
        if current_state['s'] < old_s and old_s > self.track.length / 2:
            reward += 1000.0
            terminated = True
        """
        return reward, terminated, truncated