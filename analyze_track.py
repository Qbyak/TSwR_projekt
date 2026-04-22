import numpy as np
import matplotlib.pyplot as plt

from track import create_track, create_test_track


def get_curvature(track, s, epsilon=5.0):
    p1 = track.center_line.interpolate((s - epsilon) % track.length)
    p2 = track.center_line.interpolate(s % track.length)
    p3 = track.center_line.interpolate((s + epsilon) % track.length)

    theta1 = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    theta2 = np.arctan2(p3.y - p2.y, p3.x - p2.x)

    d_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return d_theta / epsilon


def plot_track_curvature(csv_path, scale, csv_otl=None):

    track = create_track(csv_path, scale, csv_otl)
    print(f"Długość linii jazdy: {track.length:.2f} m")

    s_values = np.linspace(0, track.length, int(track.length))

    kappa_values = []
    x_values = []
    y_values = []

    for s in s_values:
        kappa = abs(get_curvature(track, s, epsilon=1.5))
        kappa_values.append(kappa)

        point = track.center_line.interpolate(s)
        x_values.append(point.x)
        y_values.append(point.y)

    # ==========================================
    # WIZUALIZACJA
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if csv_otl is None:
        fig.suptitle('Profil Krzywizny Toru: brak OTL', fontsize=16)
    else:
        fig.suptitle('Profil Krzywizny Toru: OTL', fontsize=16)


    ax1.plot(s_values, kappa_values, 'k-', linewidth=2)
    ax1.fill_between(s_values, kappa_values, 0, color='red', alpha=0.3)
    ax1.set_title('Krzywizna ($\kappa$) a dystans (s)')
    ax1.set_xlabel('Przebyty dystans s [m]')
    ax1.set_ylabel('Krzywizna $\kappa$ [1/m]')
    ax1.set_xlim(0, track.length)
    ax1.grid(True)

    scatter = ax2.scatter(x_values, y_values, c=kappa_values, cmap='jet', s=15, edgecolor='none')
    ax2.set_aspect('equal')
    ax2.set_title('Mapa zakrętów (Heatmap)')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')

    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Krzywizna $\kappa$ [1/m]')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_track_curvature('Catalunya.csv', 0.3, csv_otl='Catalunya_otl.csv')