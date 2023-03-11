import math
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
        if len(attractors[0]) == 0:
            attractors[0] = frozen
        return 0
    if frozen in attractors:
        return attractors.index(frozen) + 1

    attractors.append(frozen)
    return len(attractors)


def get_attraction_pool(config: AttractionPoolConfiguration):
    size = config.density
    x_set = np.linspace(config.x_min, config.x_max, config.density)
    y_set = np.linspace(config.y_min, config.y_max, config.density)
    cycles_map = np.zeros((size, size))

    attractors = [[]]

    for j, y in enumerate(y_set):
        for i, x in enumerate(x_set):
            point = np.array([x, y])
            points = get_points(point, config.gamma, config.sigma, config.take, config.skip)
            points = points.T
            attractor_index = get_attractor_index(points, attractors, config.take)

            cycles_map[i, j] = attractor_index

    if len(attractors[0]) == 0:
        attractors.remove([])

    return cycles_map, None, attractors


def stochastic_coupling_f(x, y, gamma_x, gamma_y, sigma):
    x_ = f(x, gamma=gamma_x) + sigma * (y - x)
    y_ = f(y, gamma=gamma_y) + sigma * (x - y)
    return x_, y_


def get_stochastic_coupling_trace(start, gammas, sigma):
    trace = np.zeros(gammas.shape)
    state = start
    for i, gamma in enumerate(gammas):
        state = stochastic_coupling_f(*state, *gamma, sigma)
        trace[i] = np.array(state)
    return trace.T


def solve_stochastic_sensitivity_matrix(function_derivative_by_point, q):
    (f11, f12), (f21, f22) = function_derivative_by_point
    a = np.array([
        [f11, f21, f11, f21],
        [f12, f22, f12, f22],
        [f11, f21, f11, f21],
        [f12, f22, f12, f22],
    ]) * np.array([
        [f11, f11, f21, f21],
        [f11, f11, f21, f21],
        [f12, f12, f22, f22],
        [f12, f12, f22, f22],
    ])
    q = q.reshape(4)

    b = np.eye(4) - a
    # det_b = np.linalg.det(b)
    b_inv = np.linalg.inv(b)

    m = q @ b_inv

    m = m.reshape((2, 2))

    return m


def get_confidence_ellipse_for_point(point, m, epsilon, p):
    w, v = np.linalg.eig(m)
    k = (-np.log(1 - p)) ** 0.5
    z = (2 * w) ** 0.5 * epsilon * k

    t = np.linspace(0, 2 * math.pi, 100)

    circle = np.array([np.cos(t), np.sin(t)])
    (v11, v12), (v21, v22) = v
    z1, z2 = (z * circle.T).T
    ellipse = point + np.array([z1 * v22 - z2 * v12, z2 * v11 - z1 * v21]).T
    return ellipse


def get_confidence_ellipse_for_equilibrium(equilibrium, sigma, epsilon, p):
    x1, x2 = equilibrium
    function_derivative_by_point = np.array([
        [f_(x1) - sigma, sigma],
        [sigma, f_(x2) - sigma]
    ])
    m = solve_stochastic_sensitivity_matrix(function_derivative_by_point, np.eye(2))

    ell = get_confidence_ellipse_for_point(equilibrium, m, epsilon, p)

    return ell


def get_confidence_ellipses_for_k_cycle(sigma, epsilon, p, k_cycle):
    k = len(k_cycle)
    fs = []
    for x in k_cycle:
        x1, x2 = x
        function_derivative_by_point = np.array([
            [f_(x1), sigma],
            [sigma, f_(x2)]
        ])
        fs.append(function_derivative_by_point)

    function_derivative_by_point = np.eye(2)
    f_prefixes = []
    for f_i in reversed(fs):
        f_prefixes.append(function_derivative_by_point)
        function_derivative_by_point = function_derivative_by_point @ f_i

    q = np.zeros((2, 2))
    qs = [np.eye(2)] * k
    for i, q_t in enumerate(qs):
        f_prefix = f_prefixes[i]
        q = q + f_prefix @ q_t @ f_prefix.T

    m = solve_stochastic_sensitivity_matrix(function_derivative_by_point, q)

    ms = [m]
    for i in range(k):
        f_t = fs[i]
        q_t = qs[i]
        m_t = ms[i]
        m_t_1 = f_t @ m_t @ f_t.T + q_t
        ms.append(m_t_1)

    ellipses = []
    for i in range(k):
        point = k_cycle[i]
        m_t = ms[i]
        ellipse_t = get_confidence_ellipse_for_point(point, m_t, epsilon, p)
        ellipses.append(ellipse_t)

    return ellipses


def get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p):
    for attractor in attractors:
        attractor = list(attractor)
        if len(attractor) == 1:
            equilibrium = attractor[0]
            ellipse = get_confidence_ellipse_for_equilibrium(equilibrium, sigma, epsilon, p)
            yield [ellipse]
        else:
            k_cycle = list(attractor)
            k = len(k_cycle)
            k_cycle = get_points(np.array(k_cycle[0]), gamma, sigma, k, 200).T
            ellipses = get_confidence_ellipses_for_k_cycle(sigma, epsilon, p, k_cycle)
            yield ellipses
