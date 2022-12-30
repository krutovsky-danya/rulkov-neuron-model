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

    for x, y1, y2 in zip(xs, ys, ys[1:]):
        if y1 * y2 < 0:
            bounds.append(x)

    bounds = np.array(bounds)

    return np.array([bounds, invert_stable_point(bounds)])
