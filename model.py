from typing import Generator, Tuple, Any

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


def get_chaotic_points_cloud(gammas, skip_points=2000, points_per_gamma=1000, x0=0, reset_x=False) -> np.ndarray:
    chaotic_points = []

    x = x0
    for j, gamma in enumerate(gammas):
        if reset_x:
            x = x0
        for _ in range(skip_points):
            x = f(x, gamma=gamma)

        for i in range(points_per_gamma):
            x = f(x, gamma=gamma)
            chaotic_points.append([gamma, x])

    return np.array(chaotic_points).T


def group_chaotic_points(chaotic_points: np.ndarray) -> Generator[Tuple[float, np.ndarray], None, Any]:
    grouped_chaotic = {}

    for gamma, x in chaotic_points.T:
        if gamma not in grouped_chaotic:
            grouped_chaotic[gamma] = list()
        grouped_chaotic[gamma].append(x)

    for gamma, xs in grouped_chaotic.items():
        yield gamma, np.array(xs)


def get_lyapunov_exponent(grouped_points):
    lyapunov_exponent = []

    for gamma, points in grouped_points:
        derivatives = f_(points)
        modules = np.abs(derivatives)
        logs = np.log(modules)
        lyapunov_exponent.append([gamma, np.mean(logs)])

    return np.array(lyapunov_exponent).T
