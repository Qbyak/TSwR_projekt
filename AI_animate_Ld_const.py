import numpy as np
import matplotlib;

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from stable_baselines3 import PPO
from racing_env_Ld_const import RacingEnv


def run_ai_animated_simulation():

    env = RacingEnv(csv_path='Catalunya.csv',scale=0.4,otl_path=None)
    model = PPO.load("ppo_model_2505_Ld.zip", env=env)
    obs, info = env.reset()
    dt = env.dt

    history_x, history_y = [], []
    history_t = []
    history_aF, history_aR = [], []
    history_vx, history_target_T = [], []
    history_Ld = []

    current_step = 0
    done = False


    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Test AI (PPO) - Dynamika, Uślizg, Prędkość i Ld', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, width_ratios=[1.5, 1])

    # Lewy panel (Tor)
    ax_track = fig.add_subplot(gs[:, 0])
    ax_track.set_aspect('equal')
    ax_track.set_title('Trasa bolidu AI')
    ax_track.set_xlabel('X [m]')
    ax_track.set_ylabel('Y [m]')

    x_center, y_center = env.track.center_line.xy
    ax_track.plot(x_center, y_center, 'k--', label='Linia centralna')

    if hasattr(env.track, 'polygon'):
        x_bound, y_bound = env.track.polygon.exterior.xy
        ax_track.plot(x_bound, y_bound, 'b-', alpha=0.5, label='Granice toru')

    car_patch = patches.Rectangle((0, 0), width=1.5, height=1.0, color='red', label='Bolid')
    ax_track.add_patch(car_patch)
    lookahead_scatter, = ax_track.plot([], [], 'ro', markersize=4, label='Cel Ld (AI)')
    history_line, = ax_track.plot([], [], 'r-', alpha=0.4)
    ax_track.legend(loc='upper right')

    ax_slip = fig.add_subplot(gs[0, 1])
    ax_slip.set_title('Kąty Uślizgu Opon')
    ax_slip.set_ylabel('Kąt uślizgu [rad]')
    ax_slip.set_xlim(0, 10)
    ax_slip.set_ylim(-0.15, 0.15)
    ax_slip.grid(True)
    line_aF, = ax_slip.plot([], [], 'g-', label=r'$\alpha_F$ (Przód)')
    line_aR, = ax_slip.plot([], [], 'm-', label=r'$\alpha_R$ (Tył)')
    ax_slip.legend(loc='upper right')

    ax_vel = fig.add_subplot(gs[2, 1])
    ax_vel.set_title('Profil Prędkości i Sterowania T')
    ax_vel.set_xlabel('Czas [s]')
    ax_vel.set_ylabel('Prędkość [m/s]', color='c')
    ax_vel.set_xlim(0, 10)
    ax_vel.set_ylim(0, 20)
    ax_vel.grid(True)


    line_vx, = ax_vel.plot([], [], 'c-', linewidth=2, label='Rzeczywista $v_x$')
    ax_vel.tick_params(axis='y', labelcolor='c')


    ax_t_pedal = ax_vel.twinx()
    ax_t_pedal.set_ylabel('Komenda Pedału T [-1, 1]', color='orange')
    ax_t_pedal.set_ylim(-1.1, 1.1)


    line_target_T, = ax_t_pedal.plot([], [], 'orange', linestyle='--', alpha=0.8, label='Pedał T (AI)')
    ax_t_pedal.tick_params(axis='y', labelcolor='orange')


    lines = [line_vx, line_target_T]
    labels = [l.get_label() for l in lines]
    ax_vel.legend(lines, labels, loc='upper right')


    ax_ld = fig.add_subplot(gs[1, 1])
    ax_ld.set_title('Dystans Widzenia (Lookahead Ld)')
    ax_ld.set_xlabel('Czas [s]')
    ax_ld.set_ylabel('Ld [m]')
    ax_ld.set_xlim(0, 10)
    ax_ld.set_ylim(1.5, 15.5)
    ax_ld.grid(True)
    line_Ld, = ax_ld.plot([], [], 'b-', linewidth=2, label='Ld (AI)')
    ax_ld.legend(loc='upper right')

    def update(frame):
        nonlocal obs, done, current_step

        STEPS_PER_FRAME = 3
        ai_Ld = 2.0
        for _ in range(STEPS_PER_FRAME):
            if not done:

                current_step += 1
                state = env.car.get_state()
                vx = state['vx']
                action, _states = model.predict(obs, deterministic=True)

                k = 1.2
                ai_Ld = np.clip(4, 1.0, 20.0)


                #ai_Ld = 12.0 + action[0] * 10.0
                target_T_ai = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    done = True


                state = env.car.get_state()
                current_time = current_step * dt

                history_t.append(current_time)
                alpha_F, alpha_R = env.car.get_slip_angles()
                history_aF.append(alpha_F)
                history_aR.append(alpha_R)
                history_vx.append(state['vx'])
                history_target_T.append(target_T_ai)
                history_Ld.append(ai_Ld)
                car_x, car_y = env.track.get_global_coords(state['s'], state['n'])
                history_x.append(car_x)
                history_y.append(car_y)

        if not history_t:
            return car_patch, lookahead_scatter, history_line, line_aF, line_aR, line_vx, line_target_T, line_Ld

        state = env.car.get_state()
        epsilon = 0.1
        p1 = env.track.center_line.interpolate((state['s'] - epsilon) % env.track.length)
        p2 = env.track.center_line.interpolate((state['s'] + epsilon) % env.track.length)
        yaw = np.arctan2(p2.y - p1.y, p2.x - p1.x) + state['mu']

        corner_x = history_x[-1] - (1.5 / 2) * np.cos(yaw) + (1.0 / 2) * np.sin(yaw)
        corner_y = history_y[-1] - (1.5 / 2) * np.sin(yaw) - (1.0 / 2) * np.cos(yaw)

        car_patch.set_xy((corner_x, corner_y))
        car_patch.set_angle(np.degrees(yaw))

        _, target_x, target_y = env.controller.compute_steering(state, env.track, ai_Ld)
        lookahead_scatter.set_data([target_x], [target_y])
        history_line.set_data(history_x, history_y)

        line_aF.set_data(history_t, history_aF)
        line_aR.set_data(history_t, history_aR)
        line_vx.set_data(history_t, history_vx)
        line_target_T.set_data(history_t, history_target_T)
        line_Ld.set_data(history_t, history_Ld)


        last_time = history_t[-1]
        if last_time > ax_slip.get_xlim()[1]:
            ax_slip.set_xlim(last_time - 5, last_time + 5)
            ax_vel.set_xlim(last_time - 5, last_time + 5)
            ax_ld.set_xlim(last_time - 5, last_time + 5)
            ax_t_pedal.set_xlim(last_time - 5, last_time + 5)

        max_alpha = max(max(np.abs(history_aF)), max(np.abs(history_aR)))
        if max_alpha > ax_slip.get_ylim()[1] * 0.9:
            ax_slip.set_ylim(-max_alpha * 1.5, max_alpha * 1.5)

        return car_patch, lookahead_scatter, history_line, line_aF, line_aR, line_vx, line_target_T, line_Ld

    ani = animation.FuncAnimation(fig, update, frames=10000, interval=25, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ai_animated_simulation()