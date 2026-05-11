import numpy as np

"""
s0 – początkowy postęp wzdłuż toru,
n0 – początkowe odchylenie boczne od linii jazdy,
mu0 – początkowy kąt względem stycznej toru,
vx0 – początkowa prędkość podłużna.
"""
class Vehicle:
    def __init__(self, s0=0.0, n0=0.0, mu0=0.0, vx0=5.0):
        # Geometria pojazdu
        self.l_F = 0.75         # [m]
        self.l_R = 0.75         # [m]
        self.L = self.l_F + self.l_R

        # Parametry masowe
        self.m = 300.0          # [kg]
        self.Iz = 120.0        # [kg m^2]
        self.g = 9.81           # [m/s^2]

        '''
        Pacejka uproszczona
        Fy = Fz * D * sin(C * arctan(B * alpha))
        B – sztywność początkowa opony
        C – kształt charakterystyki
        D – poziom maksymalnej siły / nasycenie
        '''
        self.BF = 5.0
        self.CF = 1.3
        self.DF = 1.4

        self.BR = 5.0
        self.CR = 1.3
        self.DR = 1.4


        '''
        # Napęd / opory
        # Fx = Cm*T - Cr0 - Cr2*vx^2
        Cm*T – siła napędowa albo hamująca,
        Cr0 – stały opór toczenia,
        Cr2*vx^2– opór rosnący z kwadratem prędkości.
        0.6 * 17^2 ≈ 173 N
        100 + 173 = 273 N
        273 / 300 ≈ 0.91 m/s^2 - zwalnianie auta bez gazu
        '''
        self.Cm = 3000.0        # [N]
        self.Cr0 = 100.0        # [N]
        self.Cr2 = 0.6          # [N / (m/s)^2]

        # Torque vectoring / yaw support
        # Mtv = ptv * (rt - r)
        self.ptv = 400.0       # [Nm / (rad/s)]
        '''
        Ograniczenia aktuatorów
        max_steer – maksymalny kąt skrętu kół,
        max_steer_rate – maksymalna szybkość zmiany skrętu,
        max_T – maksymalna komenda gazu,
        min_T – maksymalna komenda hamowania,
        max_T_rate – maksymalna szybkość zmiany gazu/hamowania.
        '''
        self.max_steer = 0.9       # [rad]
        self.max_steer_rate = 6.0   # [rad/s]
        self.max_T = 1.0
        self.min_T = -1.0
        self.max_T_rate = 4.0       # [1/s]

        # Stan pojazdu
        '''
        s – pozycja wzdłuż toru,
        n – odchylenie boczne od linii jazdy,
        μ – różnica kąta pojazdu względem toru,
        vx – prędkość podłużna,
        vy – prędkość poprzeczna,
        r – prędkość kątowa yaw,
        δ – aktualny kąt skrętu,
        T – komenda napędu / hamowania.
        '''
        self.s = s0
        self.n = n0
        self.mu = mu0
        self.vx = vx0
        self.vy = 0.0
        self.r = 0.0
        self.delta = 0.0
        self.T = 0.0

    # Pomocnicze funkcje
    @staticmethod #zakres od -pi do pi
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod #szybkości zmiany danej wielkości
    def _clip_with_rate(current, target, max_rate, dt):
        max_step = max_rate * dt
        return current + np.clip(target - current, -max_step, max_step)

    def _normal_loads(self):
        #statyczny rozkład obciążeń,
        #model nie uwzględnia dynamicznego transferu masy podczas hamowania,
        #przyspieszania i skręcania.
        Fz_F = self.l_R / (self.l_F + self.l_R) * self.m * self.g
        Fz_R = self.l_F / (self.l_F + self.l_R) * self.m * self.g
        return Fz_F, Fz_R

    def _slip_angles(self):
        #obliczanie kąta uślizgu przedniej i tylnej osi
        vx_safe = max(abs(self.vx), 0.5)
        alpha_F = np.arctan2(self.vy + self.l_F * self.r, vx_safe) - self.delta
        alpha_R = np.arctan2(self.vy - self.l_R * self.r, vx_safe)
        return alpha_F, alpha_R

    def get_slip_angles(self):
        return self._slip_angles()

    def _lateral_forces(self):
        # oblicza siły boczne generowane przez przednią i tylną oś
        Fz_F, Fz_R = self._normal_loads()
        alpha_F, alpha_R = self._slip_angles()

        Fy_F = -Fz_F * self.DF * np.sin(self.CF * np.arctan(self.BF * alpha_F))
        Fy_R = -Fz_R * self.DR * np.sin(self.CR * np.arctan(self.BR * alpha_R))

        return Fy_F, Fy_R, alpha_F, alpha_R

    def _longitudinal_force(self):
        # Prosty model napędu + opory
        # oblicza siłę podłużną działającą na pojazd
        Fx = self.Cm * self.T - self.Cr0 - self.Cr2 * self.vx**2
        return Fx

    def _torque_vectoring_moment(self):
        # oblicza dodatkowy moment obracający pojazd wokół osi pionowej
        # r_target = tan(delta) * vx / L
        vx_safe = max(abs(self.vx), 0.5)
        r_target = np.tan(self.delta) * vx_safe / self.L
        Mtv = self.ptv * (r_target - self.r)
        return Mtv

    '''
    model dynamiczny
    target_delta – zadany kąt skrętu z kontrolera,
    target_T – zadana komenda gazu,
    curvature – lokalna krzywizna toru,
    dt – krok czasowy symulacji.
    '''
    def update_dynamic(self, target_delta, target_T, curvature, dt=0.025):
        # dynamika aktuatorów
        target_delta = np.clip(target_delta, -self.max_steer, self.max_steer)
        target_T = np.clip(target_T, self.min_T, self.max_T)

        self.delta = self._clip_with_rate(self.delta, target_delta, self.max_steer_rate, dt)
        self.T = self._clip_with_rate(self.T, target_T, self.max_T_rate, dt)

        # siły i momenty
        Fy_F, Fy_R, alpha_F, alpha_R = self._lateral_forces()
        Fx = self._longitudinal_force()
        Mtv = self._torque_vectoring_moment()

        # równania dynamiczne
        denominator = 1.0 - self.n * curvature
        if abs(denominator) < 1e-3:
            denominator = np.sign(denominator) * 1e-3 if denominator != 0 else 1e-3
        # szybkość przemieszczania się pojazdu wzdłuż toru
        s_dot = (self.vx * np.cos(self.mu) - self.vy * np.sin(self.mu)) / denominator
        # szybkość zmiany odchylenia bocznego od linii jazdy
        n_dot = self.vx * np.sin(self.mu) + self.vy * np.cos(self.mu)
        # zmiana kąta pojazdu względem stycznej toru
        mu_dot = self.r - curvature * s_dot

        vx_dot = (Fx - Fy_F * np.sin(self.delta) + self.m * self.vy * self.r) / self.m
        vy_dot = (Fy_R + Fy_F * np.cos(self.delta) - self.m * self.vx * self.r) / self.m
        r_dot = (Fy_F * self.l_F * np.cos(self.delta) - Fy_R * self.l_R + Mtv) / self.Iz

        # całkowanie Euler
        self.s += s_dot * dt
        self.n += n_dot * dt
        self.mu += mu_dot * dt

        self.vx += vx_dot * dt
        self.vy += vy_dot * dt
        self.r += r_dot * dt

        self.mu = self._wrap_angle(self.mu)
        self.vx = max(self.vx, 0.1)

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