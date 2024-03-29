import math
from typing import Generator, Tuple, Any, Union

import numpy as np
from numba import njit, int32, float32

from infrastructure import timeit

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


class AttractionPoolConfiguration:
    def __init__(self, gamma: float, sigma, xs=(-1, 1), ys=(-1, 1), density=100, skip=2000, take=16):
        self.gamma: float = gamma
        self.sigma = sigma
        self.x_min, self.x_max = xs
        self.y_min, self.y_max = ys
        self.density = density
        self.take = take
        self.skip = skip

    def get_extent(self):
        extent = [self.x_min, self.x_max, self.y_min, self.y_max]
        return extent

    def get_cell_diameter(self):
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min

        d = min(dx, dy)

        return d / self.density


class StochasticAttractionPoolConfiguration(AttractionPoolConfiguration):
    def __init__(self, config: AttractionPoolConfiguration, epsilon, p, stochastic_count=100):
        xs = config.x_min, config.x_max
        ys = config.y_min, config.y_max
        super().__init__(config.gamma, config.sigma, xs, ys, config.density, config.skip, config.take)
        self.epsilon = epsilon
        self.p = p
        self.shift = np.random.uniform(-1, 1, 2) / (config.density ** 2)
        self.stochastic_count = stochastic_count


class StochasticResult:
    def __init__(self, config, heatmap, attractors, traces, ellipses, synchronization_indicators):
        self.config = config
        self.heatmap = heatmap
        self.attractors = attractors
        self.traces = traces
        self.ellipses = ellipses
        self.synchronization_indicators = synchronization_indicators


def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


@njit()
def f(x, alpha: float = 4.1, gamma: float = -3):
    return alpha / (1 + x ** 2) + gamma


@njit()
def f_(x, alpha: float = 4.1):
    return - (2 * alpha * x) / ((1 + x ** 2) ** 2)


@njit()
def invert_stable_point(x, alpha: float = 4.1):
    return x - alpha / (1 + x ** 2)


@njit()
def invert_stable_point_(x, alpha: float = 4.1):
    return 1 - f_(x, alpha)


def get_repeller_bounds(xs, alpha: float = 4.1) -> np.ndarray:
    ys = invert_stable_point_(xs, alpha=alpha)

    bounds = []

    for x1, x2, y1, y2 in zip(xs, xs[1:], ys, ys[1:]):
        if y1 * y2 < 0:
            bounds.append((x1 + x2) / 2)

    bounds = np.array(bounds)

    return np.array([bounds, invert_stable_point(bounds)]).T


def get_real_solutions(roots):
    real_roots = []
    for root in roots:
        if abs(root.real - root) < 1e-7:
            real_roots.append(root.real)

    return sorted(real_roots)


def get_repeller_bounds_fast(alpha=4.1):
    roots = np.roots([1, 0, 1, 2 * alpha, 1])
    return get_real_solutions(roots)


def get_stable_points_fast(alpha=4.1, gamma=0):
    roots = np.roots([1, -gamma, 1, -gamma - alpha])
    return get_real_solutions(roots)


@njit()
def get_chaotic_points_cloud(gammas, skip_points=2000, points_per_gamma=300, x0=0, reset_x=False) -> np.ndarray:
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


@njit()
def get_points_distribution(gammas: np.ndarray, xs: np.ndarray, skip, take):
    gammas_count = gammas.shape[0]
    xs_count = xs.shape[0]

    points = np.zeros((gammas_count * xs_count * take, 2))
    for gamma_i, gamma in enumerate(gammas):
        for x_i, x in enumerate(xs):
            for _ in range(skip):
                x = f(x, gamma=gamma)
            for i in range(take):
                x = f(x, gamma=gamma)
                points[gamma_i * xs_count * take + x_i * take + i] = np.array((gamma, x))

    return points.T


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
def same_coupling_f(point: np.ndarray, alfa=4.1, gamma=0.8, sigma: Union[float, np.ndarray] = 0.1) -> np.ndarray:
    move: np.ndarray = f(point, alfa, gamma)
    x, y = point
    delta = y - x
    coupling = np.zeros(point.shape)
    coupling[0] = delta
    coupling[1] = -delta
    coupling = sigma * coupling
    return move + coupling


@njit()
def get_points(point: np.ndarray, gamma: float, sigma=0.1, take=100, skip=0) -> np.ndarray:
    taken_points = get_points_2d(point, gamma, sigma, skip, take)
    return taken_points.transpose((1, 0))


@njit()
def get_points_2d(points, gamma, sigma, skip, take):
    for i in range(skip):
        points = same_coupling_f(points, gamma=gamma, sigma=sigma)

    shape = (take, *points.shape)
    taken_points = np.zeros(shape)

    taken_points[0] = points
    for i in range(1, take):
        points = same_coupling_f(points, gamma=gamma, sigma=sigma)
        taken_points[i] = points

    return taken_points


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


def get_attractor_index(origin: np.ndarray, radius: float, attractors: list, overflowed: list):
    for i, attractor in enumerate(attractors):
        for p in attractor:
            distance = np.linalg.norm(p - origin)
            if distance < radius:
                return i

    for point in overflowed:
        distance = np.linalg.norm(point - origin)
        if distance < radius:
            return -2

    return -1


def get_attractor_trace(origin: np.ndarray, gamma: float, sigma: float, radius: float, limit: int):
    cycle = []
    point = origin

    for _ in range(limit):
        cycle.append(point)
        point = same_coupling_f(point, gamma=gamma, sigma=sigma)
        distance = np.linalg.norm(origin - point)
        if distance < radius:
            break

    match = True
    k = len(cycle)
    trace = list(cycle)
    trace.append(point)
    for i in range(k + 1, limit):
        point = same_coupling_f(point, gamma=gamma, sigma=sigma)
        distance = np.linalg.norm(trace[i - k] - point)
        match = match and (distance < radius)
        trace.append(point)

    if match:
        return cycle
    else:
        return trace


def get_attractors(points, radius, limit, gamma, sigma):
    decimals = int(-np.log10(radius))

    d, h, w = points.shape

    reshaped = np.reshape(points, (d, h * w)).T
    attractor_starts = np.round(reshaped, decimals)
    unique, indexes = np.unique(attractor_starts, return_index=True, axis=0)

    attractors = []
    overflowed = []
    for i, index in enumerate(indexes):
        start = reshaped[index]
        attractor_index = get_attractor_index(start, radius, attractors, overflowed)

        if attractor_index == -1:
            attractor = get_attractor_trace(start, gamma, sigma, radius, limit)

            if len(attractor) < limit:
                attractors.append(attractor)
            else:
                overflowed = attractor

    if len(overflowed) > 0:
        attractors.append(overflowed)

    return attractors


@timeit
def get_attraction_pool(config: AttractionPoolConfiguration, dx=0, dy=0):
    x_set = np.linspace(config.x_min, config.x_max, config.density) + dx
    y_set = np.linspace(config.y_min, config.y_max, config.density) + dy

    xs = np.stack([x_set] * config.density, axis=0)
    ys = np.stack([y_set] * config.density, axis=1)

    points = np.array([xs, ys])
    taken_points = get_points_2d(points, config.gamma, config.sigma, config.skip, config.take)

    points = taken_points[-1]
    taken_points = taken_points.transpose((1, 0, 2, 3))

    x, y = taken_points
    heatmap = np.abs(x - y).mean(axis=0)

    radius = config.get_cell_diameter() / 2
    attractors = get_attractors(points, radius, config.take, config.gamma, config.sigma)

    return heatmap, attractors


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

    b_inv = np.linalg.inv(b)

    m = q @ b_inv

    m = m.reshape((2, 2))

    return m


def get_confidence_ellipse_for_point(point, m, epsilon, p):
    w, v = np.linalg.eig(m)
    k = (-np.log(1 - p)) ** 0.5

    if np.any(w < 0):
        return np.zeros(0)

    z = (2 * w) ** 0.5 * epsilon * k

    t = np.linspace(0, 2 * math.pi, 100)

    circle = np.array([np.cos(t), np.sin(t)])
    (v11, v12), (v21, v22) = v.T
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
            [f_(x1) - sigma, sigma],
            [sigma, f_(x2) - sigma]
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


def get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p, k_max=10):
    for attractor in attractors:
        attractor = list(attractor)
        if len(attractor) == 1:
            equilibrium = attractor[0]
            ellipse = get_confidence_ellipse_for_equilibrium(equilibrium, sigma, epsilon, p)
            yield [ellipse]
        else:
            k_cycle = list(attractor)
            k = len(k_cycle)

            if k > k_max:
                continue

            k_cycle = get_points_2d(np.array(k_cycle[0]), gamma, sigma, 200, k)
            ellipses = get_confidence_ellipses_for_k_cycle(sigma, epsilon, p, k_cycle)
            yield ellipses


@timeit
def get_lyapunov_exponents_2d(gamma: float, origins: np.ndarray, sigmas: np.ndarray, delta=1e-9, steps=1000):
    n_starts, n_sigmas, n_dimensions = origins.shape

    points = np.round(origins, 9).T

    sigmas_ = np.broadcast_to(sigmas, (n_starts, n_sigmas)).T
    sigmas_v = np.broadcast_to(sigmas_, points.shape)

    xs = get_points_2d(points, gamma, sigmas_v, take=steps, skip=1)

    r = (2 ** -0.5) * delta * np.ones(points.shape)
    ps = np.zeros((steps, n_sigmas, n_starts))
    xv = points + r
    for k in range(steps):
        x = xs[k]
        xv_ = same_coupling_f(xv, gamma=gamma, sigma=sigmas_v)

        d = xv_ - x
        d_norm = np.linalg.norm(d, axis=0)

        normalized_d = d / d_norm

        xv = x + normalized_d * delta

        ps[k] = d_norm / delta

    lyapunov_exponents = np.log(ps).mean(axis=0)

    return lyapunov_exponents


def get_synchronization_indicator(trace: np.ndarray):
    trace = trace.T
    d_trace = trace[1:] - trace[:-1]

    dxs, dys = d_trace.T
    zs = np.sign(dxs * dys)

    return zs


def main():
    e = (-5, 5)
    c = AttractionPoolConfiguration(-0.7, 0.1, e, e, 201, 100, 10)
    get_attraction_pool(c)


if __name__ == '__main__':
    main()
