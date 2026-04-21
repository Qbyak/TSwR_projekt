import numpy as np


class Vehicle:
    def __init__(self, s0=0.0, n0=0.0, mu0=0.0, vx0=5.0):
        # =========================
        # Geometria pojazdu
        # =========================
        self.l_F = 1.5          # [m]
        self.l_R = 1.5          # [m]
        self.L = self.l_F + self.l_R

        # =========================
        # Parametry masowe
        # =========================
        self.m = 750.0          # [kg]
        self.Iz = 1200.0        # [kg m^2]
        self.g = 9.81           # [m/s^2]

        # =========================
        # Pacejka uproszczona
        # Fy = Fz * D * sin(C * arctan(B * alpha))
        # =========================
        self.BF = 5.0
        self.CF = 1.3
        self.DF = 1.0

        self.BR = 5.0
        self.CR = 1.3
        self.DR = 1.0

        # =========================
        # Napęd / opory
        # Fx = Cm*T - Cr0 - Cr2*vx^2
        # =========================
        self.Cm = 5000.0        # [N]
        self.Cr0 = 180.0        # [N]
        self.Cr2 = 0.6          # [N / (m/s)^2]

        # =========================
        # Torque vectoring / yaw support
        # z artykułu: Mtv = ptv * (rt - r)
        # =========================
        self.ptv = 1000.0       # [Nm / (rad/s)]

        # =========================
        # Ograniczenia aktuatorów
        # =========================
        self.max_steer = 0.9       # [rad]
        self.max_steer_rate = 6.0   # [rad/s]
        self.max_T = 1.0
        self.min_T = -1.0
        self.max_T_rate = 4.0       # [1/s]

        # =========================
        # Stan pojazdu
        # =========================
        self.s = s0
        self.n = n0
        self.mu = mu0
        self.vx = vx0
        self.vy = 0.0
        self.r = 0.0
        self.delta = 0.0
        self.T = 0.0

    # ==========================================================
    # Pomocnicze funkcje
    # ==========================================================
    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _clip_with_rate(current, target, max_rate, dt):
        max_step = max_rate * dt
        return current + np.clip(target - current, -max_step, max_step)

    def _normal_loads(self):
        # Zgodnie z artykułem: statyczny rozkład obciążeń
        Fz_F = self.l_R / (self.l_F + self.l_R) * self.m * self.g
        Fz_R = self.l_F / (self.l_F + self.l_R) * self.m * self.g
        return Fz_F, Fz_R

    def _slip_angles(self):
        # Zabezpieczenie przed dzieleniem przez zero
        vx_safe = max(abs(self.vx), 0.5)

        alpha_F = np.arctan2(self.vy + self.l_F * self.r, vx_safe) - self.delta
        alpha_R = np.arctan2(self.vy - self.l_R * self.r, vx_safe)
        return alpha_F, alpha_R

    def _lateral_forces(self):
        Fz_F, Fz_R = self._normal_loads()
        alpha_F, alpha_R = self._slip_angles()

        Fy_F = -Fz_F * self.DF * np.sin(self.CF * np.arctan(self.BF * alpha_F))
        Fy_R = -Fz_R * self.DR * np.sin(self.CR * np.arctan(self.BR * alpha_R))

        return Fy_F, Fy_R, alpha_F, alpha_R

    def _longitudinal_force(self):
        # Prosty model napędu + opory
        Fx = self.Cm * self.T - self.Cr0 - self.Cr2 * self.vx**2
        return Fx

    def _torque_vectoring_moment(self):
        vx_safe = max(abs(self.vx), 0.5)
        r_target = np.tan(self.delta) * vx_safe / self.L
        Mtv = self.ptv * (r_target - self.r)
        return Mtv

    # ==========================================================
    # STARY model - można zostawić do porównań
    # ==========================================================
    def update_kinematic(self, target_delta, target_T, curvature, dt=0.025):
        self.delta = np.clip(target_delta, -self.max_steer, self.max_steer)
        self.T = np.clip(target_T, self.min_T, self.max_T)

        acceleration = self.T * 4.0
        self.vx += acceleration * dt

        self.vy = 0.0
        self.r = self.vx * np.tan(self.delta) / self.L

        denominator = 1.0 - self.n * curvature
        if abs(denominator) < 1e-3:
            denominator = np.sign(denominator) * 1e-3 if denominator != 0 else 1e-3

        s_dot = (self.vx * np.cos(self.mu) - self.vy * np.sin(self.mu)) / denominator
        n_dot = self.vx * np.sin(self.mu) + self.vy * np.cos(self.mu)
        mu_dot = self.r - curvature * s_dot

        self.s += s_dot * dt
        self.n += n_dot * dt
        self.mu += mu_dot * dt
        self.mu = self._wrap_angle(self.mu)

    # ==========================================================
    # NOWY model dynamiczny
    # ==========================================================
    def update_dynamic(self, target_delta, target_T, curvature, dt=0.025):
        # -------------------------
        # 1. dynamika aktuatorów
        # -------------------------
        target_delta = np.clip(target_delta, -self.max_steer, self.max_steer)
        target_T = np.clip(target_T, self.min_T, self.max_T)

        self.delta = self._clip_with_rate(self.delta, target_delta, self.max_steer_rate, dt)
        self.T = self._clip_with_rate(self.T, target_T, self.max_T_rate, dt)

        # -------------------------
        # 2. siły i momenty
        # -------------------------
        Fy_F, Fy_R, alpha_F, alpha_R = self._lateral_forces()
        Fx = self._longitudinal_force()
        Mtv = self._torque_vectoring_moment()

        # -------------------------
        # 3. równania dynamiczne
        # zgodnie z artykułem
        # -------------------------
        denominator = 1.0 - self.n * curvature
        if abs(denominator) < 1e-3:
            denominator = np.sign(denominator) * 1e-3 if denominator != 0 else 1e-3

        s_dot = (self.vx * np.cos(self.mu) - self.vy * np.sin(self.mu)) / denominator
        n_dot = self.vx * np.sin(self.mu) + self.vy * np.cos(self.mu)
        mu_dot = self.r - curvature * s_dot

        vx_dot = (Fx - Fy_F * np.sin(self.delta) + self.m * self.vy * self.r) / self.m
        vy_dot = (Fy_R + Fy_F * np.cos(self.delta) - self.m * self.vx * self.r) / self.m
        r_dot = (Fy_F * self.l_F * np.cos(self.delta) - Fy_R * self.l_R + Mtv) / self.Iz

        # -------------------------
        # 4. całkowanie Eulera
        # -------------------------
        self.s += s_dot * dt
        self.n += n_dot * dt
        self.mu += mu_dot * dt

        self.vx += vx_dot * dt
        self.vy += vy_dot * dt
        self.r += r_dot * dt

        # -------------------------
        # 5. zabezpieczenia numeryczne
        # -------------------------
        self.mu = self._wrap_angle(self.mu)
        self.vx = max(self.vx, 0.1)

        # opcjonalne miękkie ograniczenia stabilności
        #self.vy = np.clip(self.vy, -5.0, 5.0)
        #self.r = np.clip(self.r, -5.0, 5.0)

    def get_state(self):
        return {
            "s": self.s,
            "n": self.n,
            "mu": self.mu,
            "vx": self.vx,
            "vy": self.vy,
            "r": self.r,
            "delta": self.delta,
            "T": self.T,
        }