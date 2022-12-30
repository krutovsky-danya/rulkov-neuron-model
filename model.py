import numpy as np


def f(x, alpha: float = 4.1, gamma: float = -3):
    return alpha / (1 + x ** 2) + gamma


def f_(x, alpha: float = 4.1):
    return (2 * alpha * x) / ((1 + x ** 2) ** 2)


def invert_stable_point(x, alpha: float = 4.1):
    return x - alpha / (1 + x ** 2)


def invert_stable_point_(x, alpha: float = 4.1):
    return 1 + f_(x, alpha)


def get_repeller_bounds(xs, alpha: float = 4.1) -> np.ndarray:
    ys = invert_stable_point_(xs, alpha=alpha)

    bounds = []

    for x1, x2, y1, y2 in zip(xs, xs[1:], ys, ys[1:]):
        if y1 * y2 < 0:
            bounds.append((x1 + x2) / 2)

    bounds = np.array(bounds)

    return np.array([bounds, invert_stable_point(bounds)])


def get_chaotic_points_cloud(gammas, skip_points=2000, points_per_gamma=10000, x0=0, reset_x=False):
    chaotic_points = []
    grouped_chaotic_points = []

    x = x0
    for j, gamma in enumerate(gammas):
        if reset_x:
            x = x0
        for _ in range(skip_points):
            x = f(x, gamma=gamma)

        group = np.zeros(points_per_gamma)

        for i in range(points_per_gamma):
            x = f(x, gamma=gamma)
            chaotic_points.append([gamma, x])
            group[i] = x

        grouped_chaotic_points.append((gamma, group))

    chaotic_points = np.array(chaotic_points)

    return chaotic_points.T, grouped_chaotic_points
