import numpy as np


class PurePursuitController:
    def __init__(self, wheelbase=3.0, max_steering=0.9):
        """
        Inicjalizacja kontrolera.
        wheelbase: Rozstaw osi pojazdu (wartość L z modelu pojazdu) [m]
        max_steering: Maksymalny fizyczny kąt skrętu kół [rad] (0.5 rad to ok. 28 stopni)
        """
        self.L = wheelbase
        self.max_steering = max_steering
        self.last_delta = 0.0

    def compute_steering(self, vehicle_state, track, Ld):
        """
        Oblicza optymalny kąt skrętu na podstawie wektora stanu i dystansu Ld.
        """
        s = vehicle_state['s']
        n = vehicle_state['n']
        mu = vehicle_state['mu']

        #Lookahead Point
        s_target = (s + Ld) % track.length
        target_point = track.center_line.interpolate(s_target)
        target_x, target_y = target_point.x, target_point.y

        car_x, car_y = track.get_global_coords(s, n)
        epsilon = 0.1
        s_wrapped = s % track.length

        s_prev = (s_wrapped - epsilon) % track.length
        s_next = (s_wrapped + epsilon) % track.length

        p1 = track.center_line.interpolate(s_prev)
        p2 = track.center_line.interpolate(s_next)
        track_angle = np.arctan2(p2.y - p1.y, p2.x - p1.x)

        yaw = track_angle + mu

        angle_to_target = np.arctan2(target_y - car_y, target_x - car_x)

        alpha = angle_to_target - yaw
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Główne równanie Pure Pursuit
        delta = np.arctan((2 * self.L * np.sin(alpha)) / Ld)
        delta = np.clip(delta, -self.max_steering, self.max_steering)

        # KĄT ORAZ WSPÓŁRZĘDNE PUNKTU LOOKAHEAD
        return delta, target_x, target_y