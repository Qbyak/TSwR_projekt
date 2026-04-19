import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import pandas as pd
class Track:
    def __init__(self, center_line, track_polygon):

        self.center_line = center_line
        self.polygon = track_polygon
        self.length = center_line.length

    def is_inside(self, x, y):
        return self.polygon.contains(Point(x, y))

    def get_curvilinear_coords(self, x, y):
        """
        Konwertuje współrzędne globalne (x, y) na krzywoliniowe (s, n).
        s: progres wzdłuż ścieżki (arc-length)[cite: 85].
        n: odchylenie ortogonalne od ścieżki[cite: 85].
        """

        p = Point(x, y)
        s = self.center_line.project(p)

        # Znalezienie punktu na linii centralnej dla danego s
        target_point = self.center_line.interpolate(s)

        # Obliczenie n (odległości euklidesowej)
        distance = p.distance(target_point)

        epsilon = 0.1
        p1 = np.array(self.center_line.interpolate(max(0, s - epsilon)).coords[0])
        p2 = np.array(self.center_line.interpolate(min(self.length, s + epsilon)).coords[0])
        tangent = p2 - p1
        normal = np.array([-tangent[1], tangent[0]])  # Wektor prostopadły w lewo

        vector_to_car = np.array([x - target_point.x, y - target_point.y])

        # n > 0 (lewo), n < 0 (prawo)
        if np.dot(vector_to_car, normal) >= 0:
            n = distance
        else:
            n = -distance

        return s, n

    def get_global_coords(self, s, n):
        """
        Konwertuje współrzędne krzywoliniowe (s, n) z powrotem na globalne (x, y).
        Przydatne do wizualizacji i sprawdzania więzów toru.
        """
        s = s % self.length
        point_on_line = self.center_line.interpolate(s)

        epsilon = 0.1
        s_prev = (s - epsilon) % self.length
        s_next = (s + epsilon) % self.length

        p1 = np.array(self.center_line.interpolate(s_prev).coords[0])
        p2 = np.array(self.center_line.interpolate(s_next).coords[0])
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        # Przesunięcie o n wzdłuż wektora normalnego (kąt + 90 stopni)
        x = point_on_line.x + n * np.cos(angle + np.pi / 2)
        y = point_on_line.y + n * np.sin(angle + np.pi / 2)

        return x, y




def create_track(csv_path, scale):
    df = pd.read_csv(csv_path)

    # Pobranie danych
    x = df['x_m'].values * scale
    y = df['y_m'].values * scale
    w_left = df['w_tr_left_m'].values * (scale*3)
    w_right = df['w_tr_right_m'].values * (scale*3)

    left_boundary = []
    right_boundary = []

    n_points = len(x)
    for i in range(n_points):
        # Wyznaczanie punktu następnego i poprzedniego do wektora stycznego
        idx_prev = (i - 1) % n_points
        idx_next = (i + 1) % n_points

        # Wektor styczny
        dx = x[idx_next] - x[idx_prev]
        dy = y[idx_next] - y[idx_prev]
        length = np.hypot(dx, dy)

        if length == 0:
            continue

        # Znormalizowany wektor prostopadły (normalny) skierowany w lewo
        nx = -dy / length
        ny = dx / length

        # Obliczanie współrzędnych lewej i prawej granicy
        x_l = x[i] + nx * w_left[i]
        y_l = y[i] + ny * w_left[i]
        left_boundary.append((x_l, y_l))

        x_r = x[i] - nx * w_right[i]
        y_r = y[i] - ny * w_right[i]
        right_boundary.append((x_r, y_r))



    center_line = LineString(list(zip(x, y)))
    track_polygon = Polygon(left_boundary + right_boundary[::-1])

    return Track(center_line, track_polygon)

def plot_track(track):
    """
    Rysuje tor wyścigowy korzystając z matplotlib.
    Pobiera geometrię bezpośrednio ze zmodyfikowanego obiektu Track
    (uwzględniającego zmienną szerokość).
    """
    fig, ax = plt.subplots(figsize=(10, 6))


    x_center, y_center = track.center_line.xy
    ax.plot(x_center, y_center, 'k--', label='Linia centralna')


    x_bound, y_bound = track.polygon.exterior.xy
    ax.plot(x_bound, y_bound, 'b-', label='Granice ')
    for interior in track.polygon.interiors:
        x_in, y_in = interior.xy
        ax.plot(x_in, y_in, 'b-')

    ax.plot(x_center[0], y_center[0], 'go', markersize=8, label='Start')
    ax.set_aspect('equal')
    ax.set_title('Wizualizacja Toru')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    plt.grid(True)
    plt.show()


def create_test_track(width=3.0):
    """Tworzy przykładowy tor w kształcie stadionu (owal)."""
    # Proste odcinki i półokręgi
    points = []
    # Dolna prosta
    for x in np.linspace(0, 50, 10): points.append((x, 0))
    # Prawy łuk
    for a in np.linspace(-np.pi / 2, np.pi / 2, 10): points.append((50 + 15 * np.cos(a), 15 + 15 * np.sin(a)))
    # Górna prosta
    for x in np.linspace(50, 0, 10): points.append((x, 30))
    # Lewy łuk
    for a in np.linspace(np.pi / 2, 3 / 2 * np.pi, 10): points.append((0 + 15 * np.cos(a), 15 + 15 * np.sin(a)))
    # Zamknięcie pętli
    points.append(points[0])
    center_line = LineString(points)

    track_polygon = center_line.buffer(width / 2.0, cap_style=1, join_style=1)

    # Zwracamy kompletny obiekt Track
    return Track(center_line, track_polygon)

if __name__ == "__main__":
    track = create_track('Catalunya.csv', 0.2)
    print(f"Tor utworzony. Długość: {track.length:.2f}m")

    plot_track(track)