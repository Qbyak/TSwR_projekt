import numpy as np


class Vehicle:
    def __init__(self, s0=0.0, n0=0.0, mu0=0.0, vx0=5.0):
        # Parametry wymiarowe pojazdu (w oparciu o bolid FS)
        self.l_F = 1.5  # Odległość od środka ciężkości do przedniej osi [m] [cite: 91]
        self.l_R = 1.5  # Odległość od środka ciężkości do tylnej osi [m] [cite: 91]
        self.L = self.l_F + self.l_R  # Całkowity rozstaw osi [m]

        # Inicjalizacja pełnego wektora stanu opisanego w publikacji
        self.s = s0  # Progres wzdłuż toru
        self.n = n0  # Odchylenie poprzeczne
        self.mu = mu0  # Lokalne odchylenie kątowe względem toru
        self.vx = vx0  # Prędkość wzdłużna
        self.vy = 0.0  # Prędkość poprzeczna (w M1.2 zakładamy idealną przyczepność: vy = 0)
        self.r = 0.0  # Prędkość obrotowa (yaw rate)
        self.delta = 0.0  # Kąt skrętu kół
        self.T = 0.0  # Komenda przepustnicy/hamulca [-1, 1]

    def update_kinematic(self, target_delta, target_T, curvature, dt=0.025):
        """
        Aktualizuje stan modelu używając uproszczonych równań kinematycznych.
        Krok czasowy dt = 0.025s odpowiada 25ms używanym w publikacji[cite: 276].
        """
        # 1. Prosta dynamika aktuatorów
        self.delta = target_delta
        self.T = target_T

        # Prosta symulacja przyspieszenia (w modelu docelowym zastąpimy to siłą Fx)
        acceleration = self.T * 4.0  # max przyspieszenie np. 4 m/s^2
        self.vx += acceleration * dt

        # 2. Zależności kinematyczne (brak uślizgu)
        self.vy = 0.0
        self.r = self.vx * np.tan(self.delta) / self.L

        # 3. Równania różniczkowe we współrzędnych krzywoliniowych z publikacji [cite: 96, 97]
        # Mianownik równania na s_dot: (1 - n * k(s))
        denominator = 1.0 - self.n * curvature

        # Zabezpieczenie przed osobliwością
        if abs(denominator) < 1e-3:
            denominator = 1e-3 * np.sign(denominator)

        s_dot = (self.vx * np.cos(self.mu) - self.vy * np.sin(self.mu)) / denominator # [cite: 96]
        n_dot = self.vx * np.sin(self.mu) + self.vy * np.cos(self.mu) # [cite: 96]
        mu_dot = self.r - curvature * s_dot # [cite: 97]

        # euler
        self.s += s_dot * dt
        self.n += n_dot * dt
        self.mu += mu_dot * dt

        # Normalizacja kąta mu do przedziału [-pi, pi]
        self.mu = (self.mu + np.pi) % (2 * np.pi) - np.pi

    def get_state(self):
        """Zwraca kluczowe elementy wektora stanu do logowania."""
        return {'s': self.s, 'n': self.n, 'mu': self.mu, 'vx': self.vx}