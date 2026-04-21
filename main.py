import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from track import create_test_track, create_track
from vehicle import Vehicle
from controller import PurePursuitController


def get_curvature(track, s, epsilon=0.5):
    p1 = track.center_line.interpolate((s - epsilon) % track.length)
    p2 = track.center_line.interpolate(s % track.length)
    p3 = track.center_line.interpolate((s + epsilon) % track.length)

    theta1 = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    theta2 = np.arctan2(p3.y - p2.y, p3.x - p2.x)

    d_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return d_theta / epsilon


def run_animated_simulation():

    track = create_track('Catalunya.csv',0.1)
    car = Vehicle(s0=0.0, n0=0.0, mu0=0.0, vx0=8.0)
    controller = PurePursuitController(wheelbase=3.0, max_steering=0.9)

    dt = 0.025
    Ld = 1.5

    history_x, history_y = [], []


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_title('Test Pure Pursuit - Animacja (Milestone 1.3)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')


    x_center, y_center = track.center_line.xy
    ax.plot(x_center, y_center, 'k--', label='Linia centralna')
    x_bound, y_bound = track.polygon.exterior.xy
    ax.plot(x_bound, y_bound, 'b-', alpha=0.5, label='Granice toru')


    car_patch = patches.Rectangle((0, 0), width=3.0, height=1.5, color='red', alpha=0.8, label='Bolid')
    ax.add_patch(car_patch)

    lookahead_scatter, = ax.plot([], [], 'ro', markersize=6, label='Punkt Lookahead')
    history_line, = ax.plot([], [], 'r-', alpha=0.4)

    ax.legend(loc='upper right')

    def update(frame):
        state = car.get_state()
        curvature = get_curvature(track, state['s'])
        vx = state["vx"]
        kappa_now = abs(curvature)
        kappa_1 = abs(get_curvature(track, state['s'] + 2.0))
        kappa_2 = abs(get_curvature(track, state['s'] + 4.0))
        kappa = max(kappa_now, kappa_1, kappa_2)
        # 1. Ciągłe Ld: większe na prostych, mniejsze w zakrętach,
        # ale też trochę rośnie z prędkością
        Ld_current = 1.7 + 0.06 * vx - 3.0 * kappa
        # lokalna poprawka tylko na bardzo ciasne zakręty
        if kappa > 0.09:
            Ld_current -= 0.3
        Ld_current = np.clip(Ld_current, 1.1, 2.6)

        delta, target_x, target_y = controller.compute_steering(state, track, Ld_current)
        delta = np.clip(delta, -0.45, 0.45)

        # 2. Docelowa prędkość z krzywizny
        # im większa krzywizna, tym mniejsza prędkość
        v_ref = np.sqrt(1.5 / (kappa + 1e-3))
        v_ref = np.clip(v_ref, 2.0, 10.0)

        # 3. Prosty regulator prędkości
        target_T = 0.15 * (v_ref - vx)
        target_T = np.clip(target_T, -0.45, 0.40)

        car.update_dynamic(target_delta=delta, target_T=target_T, curvature=curvature, dt=dt)

        car_x, car_y = track.get_global_coords(car.s, car.n)
        history_x.append(car_x)
        history_y.append(car_y)

        epsilon = 0.1
        p1 = track.center_line.interpolate((car.s - epsilon) % track.length)
        p2 = track.center_line.interpolate((car.s + epsilon) % track.length)
        track_angle = np.arctan2(p2.y - p1.y, p2.x - p1.x)
        yaw = track_angle + car.mu

        corner_x = car_x - (3.0 / 2) * np.cos(yaw) + (1.5 / 2) * np.sin(yaw)
        corner_y = car_y - (3.0 / 2) * np.sin(yaw) - (1.5 / 2) * np.cos(yaw)

        car_patch.set_xy((corner_x, corner_y))
        car_patch.set_angle(np.degrees(yaw))

        lookahead_scatter.set_data([target_x], [target_y])
        history_line.set_data(history_x, history_y)

        return car_patch, lookahead_scatter, history_line

    ani = animation.FuncAnimation(fig, update, frames=800, interval=25, blit=False)
    plt.show()


if __name__ == "__main__":
    run_animated_simulation()