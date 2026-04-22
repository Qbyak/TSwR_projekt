import numpy as np
import matplotlib;

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from track import create_test_track, create_track
from vehicle import Vehicle
from controller import PurePursuitController


def get_curvature(track, s, epsilon=1.5):
    p1 = track.center_line.interpolate((s - epsilon) % track.length)
    p2 = track.center_line.interpolate(s % track.length)
    p3 = track.center_line.interpolate((s + epsilon) % track.length)

    theta1 = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    theta2 = np.arctan2(p3.y - p2.y, p3.x - p2.x)

    d_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return d_theta / epsilon


def run_animated_simulation():

    track = create_track('Catalunya.csv', 0.3, 'Catalunya_otl.csv')
    car = Vehicle(s0=0.0, n0=0.0, mu0=0.0, vx0=17.0)
    controller = PurePursuitController(wheelbase=1.5, max_steering=0.9)

    dt = 0.025

    history_x, history_y = [], []
    history_t = []
    history_aF, history_aR = [], []
    history_vx, history_vref = [], []

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('Test Pure Pursuit - Dynamika, Uślizg i Prędkość (Milestone 2.1)')

    # Lewy duży panel (Tor)
    ax_track = plt.subplot(1, 2, 1)
    ax_track.set_aspect('equal')
    ax_track.set_title('Trasa bolidu')
    ax_track.set_xlabel('X [m]')
    ax_track.set_ylabel('Y [m]')

    x_center, y_center = track.center_line.xy
    ax_track.plot(x_center, y_center, 'k--', label='Linia centralna')

    if hasattr(track, 'polygon'):
        x_bound, y_bound = track.polygon.exterior.xy
        ax_track.plot(x_bound, y_bound, 'b-', alpha=0.5, label='Granice toru')

    car_patch = patches.Rectangle((0, 0), width=1.5, height=1.0, color='red', label='Bolid')
    ax_track.add_patch(car_patch)
    lookahead_scatter, = ax_track.plot([], [], 'ro', markersize=4, label='Cel Ld')
    history_line, = ax_track.plot([], [], 'r-', alpha=0.4)
    ax_track.legend(loc='upper right')

    # Prawy GÓRNY panel (Uślizgi)
    ax_slip = plt.subplot(2, 2, 2)
    ax_slip.set_title('Kąty Uślizgu Opon')
    ax_slip.set_ylabel('Kąt uślizgu [rad]')
    ax_slip.set_xlim(0, 10)
    ax_slip.set_ylim(-0.15, 0.15)
    ax_slip.grid(True)
    line_aF, = ax_slip.plot([], [], 'g-', label=r'$\alpha_F$ (Przód)')
    line_aR, = ax_slip.plot([], [], 'm-', label=r'$\alpha_R$ (Tył)')
    ax_slip.legend(loc='upper right')

    # Prawy DOLNY panel (Prędkość)
    ax_vel = plt.subplot(2, 2, 4)
    ax_vel.set_title('Profil Prędkości')
    ax_vel.set_xlabel('Czas [s]')
    ax_vel.set_ylabel('Prędkość [m/s]')
    ax_vel.set_xlim(0, 10)
    ax_vel.set_ylim(0, 32)
    ax_vel.grid(True)
    line_vx, = ax_vel.plot([], [], 'c-', linewidth=2, label='Rzeczywista $v_x$')
    line_vref, = ax_vel.plot([], [], 'k--', alpha=0.7, label='Docelowa $v_{ref}$')
    ax_vel.legend(loc='upper right')

    # ==========================================
    # PĘTLA SYMULACYJNA
    # ==========================================
    def update(frame):
        state = car.get_state()
        vx = state["vx"]

        # RADAR
        curvature_now = get_curvature(track, state['s'], epsilon=1.5)
        kappa_now = abs(curvature_now)

        L_brake = 1.5 * vx  # ok 1.s do przodu
        curvature_ahead = get_curvature(track, state['s'] + L_brake, epsilon=1.5)
        kappa_ahead = abs(curvature_ahead)

        # Sterowanie Ld
        Ld_current = 1.5 + 0.8 * vx - 1.2 * kappa_now
        Ld_current = max(Ld_current, 1.5)
        delta_raw, target_x, target_y = controller.compute_steering(state, track, Ld_current)
        alpha = 0.3
        delta = (1.0 - alpha) * controller.last_delta + alpha * delta_raw
        controller.last_delta = delta

        kappa_max = max(kappa_now, kappa_ahead)
        v_ref = np.sqrt(10.0 / (kappa_max + 1e-3))
        v_ref = min(v_ref, 17.0)

        target_T = 1.5 * (v_ref - vx)
        target_T = np.clip(target_T, -1.0, 1.0)

        car.update_dynamic(target_delta=delta, target_T=target_T, curvature=curvature_now, dt=dt)


        current_time = frame * dt
        history_t.append(current_time)

        vy, r = car.vy, car.r
        l_F, l_R = 1.5, 1.5
        vx_safe = max(abs(car.vx), 0.5)
        alpha_F = np.arctan2(vy + l_F * r, vx_safe) - car.delta
        alpha_R = np.arctan2(vy - l_R * r, vx_safe)
        history_aF.append(alpha_F)
        history_aR.append(alpha_R)

        # Prędkość
        history_vx.append(car.vx)
        history_vref.append(v_ref)

        # ================== RYSOWANIE ==================
        car_x, car_y = track.get_global_coords(car.s, car.n)
        history_x.append(car_x)
        history_y.append(car_y)

        epsilon = 0.1
        p1 = track.center_line.interpolate((car.s - epsilon) % track.length)
        p2 = track.center_line.interpolate((car.s + epsilon) % track.length)
        yaw = np.arctan2(p2.y - p1.y, p2.x - p1.x) + car.mu

        corner_x = car_x - (1.5 / 2) * np.cos(yaw) + (1.0 / 2) * np.sin(yaw)
        corner_y = car_y - (1.5 / 2) * np.sin(yaw) - (1.0 / 2) * np.cos(yaw)

        car_patch.set_xy((corner_x, corner_y))
        car_patch.set_angle(np.degrees(yaw))
        lookahead_scatter.set_data([target_x], [target_y])
        history_line.set_data(history_x, history_y)

        line_aF.set_data(history_t, history_aF)
        line_aR.set_data(history_t, history_aR)

        line_vx.set_data(history_t, history_vx)
        line_vref.set_data(history_t, history_vref)

        if current_time > ax_slip.get_xlim()[1]:
            ax_slip.set_xlim(current_time - 5, current_time + 5)
            ax_vel.set_xlim(current_time - 5, current_time + 5)

        # Dynamiczna oś Y dla uślizgów
        max_alpha = max(max(np.abs(history_aF)), max(np.abs(history_aR)))
        if max_alpha > ax_slip.get_ylim()[1] * 0.9:
            ax_slip.set_ylim(-max_alpha * 1.5, max_alpha * 1.5)

        return car_patch, lookahead_scatter, history_line, line_aF, line_aR, line_vx, line_vref

    ani = animation.FuncAnimation(fig, update, frames=5000, interval=25, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_animated_simulation()