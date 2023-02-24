from typing import Generator, Tuple, Any

import numpy as np
from numba import njit, int32, float32

spec = [
    ('gamma', float32),
    ('sigma', float32),
    ('x_min', float32),
    ('x_max', float32),
    ('y_min', float32),
    ('y_max', float32),
    ('density', int32),
    ('skip', int32),
    ('take', int32),
]


# @jitclass(spec)
class AttractionPoolConfiguration:
    def __init__(self, gamma: float, sigma, xs=(-1, 1), ys=(-1, 1), density=100, skip=2000, take=16):
        self.gamma: float = gamma
        self.sigma = sigma
        self.x_min, self.x_max = xs
        self.y_min, self.y_max = ys
        self.density = density
        self.take = take
        self.skip = skip


@njit()
def f(x, alpha: float = 4.1, gamma: float = -3):
    return alpha / (1 + x ** 2) + gamma


@njit()
def f_(x, alpha: float = 4.1):
    return (2 * alpha * x) / ((1 + x ** 2) ** 2)


@njit()
def invert_stable_point(x, alpha: float = 4.1):
    return x - alpha / (1 + x ** 2)


@njit()
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


@njit()
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


def get_lyapunov_exponent(grouped_points) -> np.ndarray:
    lyapunov_exponent = []

    for gamma, points in grouped_points:
        derivatives = f_(points)
        modules = np.abs(derivatives)
        logs = np.log(modules)
        lyapunov_exponent.append([gamma, np.mean(logs)])

    return np.array(lyapunov_exponent).T


@njit()
def get_x_sequence(gamma: float, x0: float, steps_count=100, skip_count=0):
    xs = np.zeros(steps_count)

    x = x0
    for i in range(skip_count):
        x = f(x, gamma=gamma)

    xs[0] = x
    for i in range(1, steps_count):
        xs[i] = x = f(x, gamma=gamma)

    return xs


@njit()
def get_leader(xs):
    steps_count = len(xs)

    x = np.zeros(steps_count * 2)
    y = np.zeros(steps_count * 2)

    y[0::2] = xs
    y[1::2] = xs

    x[0] = xs[0]
    x[1:] = y[:-1]

    return x, y


COUPLING = np.array([[-1, 1], [1, -1]])


@njit()
def same_coupling_f(point: np.ndarray, alfa=4.1, gamma=0.8, sigma=0.1) -> np.ndarray:
    move: np.ndarray = f(point, alfa, gamma)
    x, y = point
    delta = y - x
    return move + sigma * np.array([delta, -delta])


@njit()
def get_points(point: np.ndarray, gamma: float, sigma=0.1, steps_count=100, skip_count=0) -> np.ndarray:
    for i in range(skip_count):
        point = same_coupling_f(point, gamma=gamma, sigma=sigma)

    trace = np.zeros((steps_count, 2))

    trace[0] = point
    for i in range(1, steps_count):
        trace[i] = point = same_coupling_f(point, gamma=gamma, sigma=sigma)

    return trace.T


@njit()
def get_points_by_sigmas(origin: np.ndarray, gamma: float, sigmas: np.ndarray,
                         restart=False, steps_count=20, skip_count=500) -> np.ndarray:
    trajectories = np.zeros((len(sigmas), 2, steps_count))

    point = origin
    for i, sigma in enumerate(sigmas):
        points = get_points(point, gamma, sigma, steps_count, skip_count)
        point = origin if restart else points[:, -1]
        trajectories[i] = points

    return trajectories


@njit()
def get_parametrized_points(sigmas, points: np.ndarray, dim=0):
    sigmas_count, _, xs_count = points.shape
    parametrized = np.zeros((sigmas_count * xs_count, 2))
    for i, (trace, sigma) in enumerate(zip(points, sigmas)):
        xs = trace[dim]
        for j, x in enumerate(xs):
            parametrized[i * xs_count + j] = np.array([sigma, x])

    return parametrized.T


def get_attractor_index(points, attractors: list, max_points):
    rounded = np.around(points, 5)
    frozen = frozenset(map(tuple, rounded.tolist()))

    if len(frozen) >= 0.8 * max_points:
        return 0
    if frozen in attractors:
        return attractors.index(frozen) + 1

    attractors.append(frozen)
    return len(attractors)


def get_attraction_pool(config: AttractionPoolConfiguration):
    size = config.density
    take = config.take
    x_set = np.linspace(config.x_min, config.x_max, config.density)
    y_set = np.linspace(config.y_min, config.y_max, config.density)
    cycles_map = np.zeros((size, size))
    last_points = np.zeros((size, size, take, 2))

    attractors = []

    for j, y in enumerate(y_set):
        for i, x in enumerate(x_set):
            point = np.array([x, y])
            points = get_points(point, config.gamma, config.sigma, config.take, config.skip)
            points = points.T
            attractor_index = get_attractor_index(points, attractors, config.take)
            last_points[i, j] = points

            cycles_map[i, j] = attractor_index

    return cycles_map, last_points, attractors
