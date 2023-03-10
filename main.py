import math

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import time

import animation
import model
import plot


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def show_stable_points(show_graphics, xs):
    gammas = model.invert_stable_point(xs)
    if show_graphics:
        plot.show_stable_points(xs, gammas)
    bounds = model.get_repeller_bounds(xs)
    print(bounds)
    if show_graphics:
        plot.show_stable_points(xs, gammas, bounds)
    return bounds


def show_where_is_repeller(bounds, show_graphics, xs):
    ys_ = model.invert_stable_point_(xs)
    if show_graphics:
        plot.show_repeller_position(xs, ys_, bounds)


def show_bifurcation(show_graphics, x_bounds, x_min):
    repeller_xs = np.linspace(*x_bounds)
    repeller_ys = model.invert_stable_point(repeller_xs)
    attractor_xs = np.linspace(x_min, min(*x_bounds))
    attractor_ys = model.invert_stable_point(attractor_xs)
    gamma_bound = model.invert_stable_point(x_bounds[1])
    chaotic_gammas = np.linspace(gamma_bound, 1, 600)
    chaotic_points = model.get_chaotic_points_cloud(chaotic_gammas)
    if show_graphics:
        plot.show_bifurcation_diagram((attractor_ys, attractor_xs), (repeller_ys, repeller_xs), chaotic_points)
    return (attractor_xs, attractor_ys), chaotic_points


def show_lyapunov(chaotic_points, attractor, show_graphics=False):
    grouped_chaotic_points = list(model.group_chaotic_points(chaotic_points))

    chaotic_lyapunov_exponent = model.get_lyapunov_exponent(grouped_chaotic_points)
    attractor_lyapunov_exponent = model.get_lyapunov_exponent(attractor)

    if show_graphics:
        plot.show_lyapunov_exponent(chaotic_lyapunov_exponent, attractor_lyapunov_exponent)


def get_sequence_generator(gamma, steps_count=100, skip_count=0):
    def func(x):
        return model.get_x_sequence(gamma, x, steps_count, skip_count)

    return func


def show_portraits(gamma, *starts, steps_count=100, skip_count=0):
    sequence_func = get_sequence_generator(gamma, steps_count, skip_count)
    sequences = list(map(sequence_func, starts))
    leaders = map(model.get_leader, sequences)

    x_min = min(map(min, sequences)) - 0.5
    x_max = max(map(max, sequences)) + 0.5

    xs = np.linspace(x_min, x_max)
    ys = model.f(xs, gamma=gamma)

    plot.show_phase_portraits(gamma, (xs, ys), sequences, leaders)


def show_several_phase_portraits(show_graphics):
    if not show_graphics:
        return
    show_portraits(-3.3, -3.1, -0.8, 0)
    show_portraits(-3, -3.1, -1.09, 0.4)
    show_portraits(-3.3, -3.1, -1.09, 0.5)
    # show_portraits(-3.55, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.4875, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.494925, 0.5, steps_count=24, skip_count=650)
    # show_portraits(-3.4949, 0.5, steps_count=20, skip_count=1150)
    # show_portraits(-3.3, 0.5, steps_count=60, skip_count=10000)


def show_1d_graphics(show_graphics=False):
    x_min = -4

    xs = np.linspace(x_min, 1, 500)
    bounds = show_stable_points(show_graphics, xs)

    show_where_is_repeller(bounds, show_graphics, xs)

    attractor, chaotic_points = show_bifurcation(show_graphics, bounds[0], x_min)

    show_lyapunov(chaotic_points, zip(*attractor[::-1]), show_graphics)

    show_several_phase_portraits(show_graphics)


def show_bifurcation_diagram_2d(gamma: float, sigmas):
    num = 7
    edges = (-5, 5)
    config = model.AttractionPoolConfiguration(gamma, sigmas.max(), edges, edges, num, 500, 100)
    _, _, attractors = model.get_attraction_pool(config)
    points_sets = []

    for restarting in [True, False]:
        for attractor in attractors:
            origin = np.array(list(attractor)[0])
            points_restarting = model.get_points_by_sigmas(origin, gamma, sigmas, restart=restarting)
            points_set = model.get_parametrized_points(sigmas, points_restarting)

            points_sets.append(points_set)

    plot.show_bifurcation_diagram_2d(gamma, points_sets)


def separate_points(last_points, cpi, steps_count):
    lined = np.reshape(last_points, (cpi * cpi * steps_count, 2))
    rounded = np.around(lined, 5)
    points = np.unique(rounded, axis=0)

    final_points = points.T

    co_mask = final_points[0] == final_points[1]

    return final_points[:, co_mask], final_points[:, ~co_mask]


def show_attraction_pool(config: model.AttractionPoolConfiguration, filename=None, show=True):
    gamma = config.gamma
    sigma = config.sigma

    heatmap, _, attractors = model.get_attraction_pool(config)

    extent = [config.x_min, config.x_max, config.y_min, config.y_max]

    for i in range(len(attractors)):
        listed = list(attractors[i])
        attractors[i] = np.array(listed).T

    traces = []
    for attractor in attractors:
        origin = np.array(attractor[:, 0])
        trace = model.get_points(origin, gamma, sigma, 200)
        traces.append(trace)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig: plt.Figure = fig

    fig.set_size_inches(14, 7)
    fig.suptitle(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f}$", size=20)

    plot.plot_attraction_pool(fig, ax1, heatmap, extent)
    plot.plot_attractors(fig, ax1, attractors)

    plot.plot_attraction_pool(fig, ax2, heatmap, extent)
    plot.plot_attractors(fig, ax2, attractors)
    for trace in traces:
        ax2.plot(*trace, '-')

    plt.show()


def show_deterministic_2d_graphics(gamma, special_sigmas, edges):
    sigmas = np.linspace(0.48, 0, 201)
    show_bifurcation_diagram_2d(gamma, sigmas)

    config = model.AttractionPoolConfiguration(gamma, 0, edges, edges, 50, 500, 100)

    for sigma in special_sigmas:
        config.sigma = sigma
        show_attraction_pool(config)


def show_2d_graphics():
    show_deterministic_2d_graphics(-0.7, [0.05, 0.1, 0.3, 0.45], (-5, 5))

    show_deterministic_2d_graphics(0.7, [0.04, 0.2], (-3, 8))


def show_stochastic_2d_graphics(gamma=-0.7, sigma=0.05, epsilon=0.1, border=(-5, 5)):
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, 20, 500, 20)

    extent = [config.x_min, config.x_max, config.y_min, config.y_max]
    heatmap, _, attractors = model.get_attraction_pool(config)

    s = np.random.normal(0, 1, (300, 2))

    stochastic_gammas = gamma + epsilon * s
    stochastic_traces = []
    for attractor in attractors:
        f = list(attractor)
        trace = model.get_stochastic_coupling_trace(f[0], stochastic_gammas, sigma)
        stochastic_traces.append(trace)

    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(14, 7)
    fig.suptitle(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}$", size=15)

    plot.plot_attraction_pool(fig, ax, heatmap, extent)

    for stochastic_trace in stochastic_traces:
        ax.plot(*stochastic_trace, '.')

    plt.show()


def build_attraction_pool_movie(show=True):
    s = 0.1
    filenames = []

    for i, g in enumerate(np.linspace(-1, -1.2, 201)):
        config = model.AttractionPoolConfiguration(g, s, (-2, 6), (-2, 6), 300)
        filename = f'images/image_{i}.png'
        show_attraction_pool(config, filename=filename, show=show)
        filenames.append(filename)

    animation.build_video("animations/from_gamma.mov", filenames)


def make_cool_zooming_movie():
    s = 0.1
    g = -1.1211
    filenames = []

    frames = 50
    b1s = -np.logspace(np.log(1), np.log(0.9775), frames, base=np.e)
    b2s = np.logspace(np.log(1 + 1), np.log(-0.9776 + 1), frames, base=np.e) - 1

    for i, (b1, b2) in enumerate(zip(b1s, b2s)):
        config = model.AttractionPoolConfiguration(g, s, (b1, b2), (b1, b2), 300)
        filename = f'images/image_zoom_{i}.png'
        show_attraction_pool(config, filename=filename, show=True)
        filenames.append(filename)

    animation.build_video("animations/zooming.mov", filenames)


def make_cool_pools():
    config = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), 200)
    show_attraction_pool(config, filename='big_special.png', show=True)

    config.x_min, config.x_max, config.y_min, config.y_max = -1, -0.99, -1, -0.99
    show_attraction_pool(config, filename='super_small_special.png', show=True)

    config.x_min, config.x_max, config.y_min, config.y_max = -100, -1, -100, -1
    show_attraction_pool(config, filename='very_big_special.png', show=True)


def make_circle_pool():
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-2.5, 5), (-2.5, 5), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-1, 4), (-1, 4), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 3.5), (-1, 1), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-1, 1), density=300)
    conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-0.5, 0.5), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1.6, 2), (-0.2, 0.2), density=500)
    show_attraction_pool(conf)


def solve_stochastic_sensitivity_matrix(f, q):
    (f11, f12), (f21, f22) = f
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


def show_confidence_ellipse_for_equilibrium(gamma, sigma, epsilon, border, p=0.99):
    origin = np.zeros(2)
    equilibrium = model.get_points(origin, gamma, sigma, 1, 1000)[:, 0]
    stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (1000, 2))
    stochastic_trace = model.get_stochastic_coupling_trace(equilibrium, stochastic_gammas, sigma)

    x1, x2 = equilibrium
    f = np.array([
        [model.f_(x1) - sigma, sigma],
        [sigma, model.f_(x2) - sigma]
    ])

    m = solve_stochastic_sensitivity_matrix(f, np.eye(2))

    ell = get_confidence_ellipse_for_point(equilibrium, m, epsilon, p)

    plt.title(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$", size=15)
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20, rotation=0)
    plt.plot(*stochastic_trace, '.', markersize=1, label="Stochastic points")
    plt.plot(*ell.T, label='Confidence ellipsis')
    plt.xlim(*border)
    plt.ylim(*border)
    plt.legend()
    plt.show()


def get_confidence_ellipses_for_k_cycle(gamma, sigma, epsilon, p, k_cycle):
    k = len(k_cycle)
    fs = []
    for x in k_cycle:
        x1, x2 = x
        f = np.array([
            [model.f_(x1), sigma],
            [sigma, model.f_(x2)]
        ])
        fs.append(f)

    f = np.eye(2)
    f_prefixes = []
    for f_i in reversed(fs):
        f_prefixes.append(f)
        f = f @ f_i

    q = np.zeros((2, 2))
    qs = [np.eye(2)] * k
    for i, q_t in enumerate(qs):
        f_prefix = f_prefixes[i]
        q = q + f_prefix @ q_t @ f_prefix.T

    m = solve_stochastic_sensitivity_matrix(f, q)

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


def show_confidence_ellipses_for_k_cycle(gamma, sigma, epsilon, border, p):
    conf = model.AttractionPoolConfiguration(gamma, sigma, border, border, density=5)
    _, _, attractors = model.get_attraction_pool(conf)
    k_cycles = []
    for attractor in attractors:
        if len(attractor) != 1:
            k_cycles.append(attractor)

    groups_of_ellipses = []
    stochastic_traces = []
    for k_cycle in k_cycles:
        k_cycle = list(k_cycle)
        k = len(k_cycle)
        k_cycle = model.get_points(np.array(list(k_cycle[0])), gamma, sigma, k, 200).T
        ellipses = get_confidence_ellipses_for_k_cycle(gamma, sigma, epsilon, p, k_cycle)
        groups_of_ellipses.append(ellipses)

        stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (1000, 2))
        stochastic_trace = model.get_stochastic_coupling_trace(k_cycle[0], stochastic_gammas, sigma)

        stochastic_traces.append(stochastic_trace)

    for t in range(len(k_cycles)):
        ellipses = groups_of_ellipses[t]
        stochastic_trace = stochastic_traces[t]
        for i, ellipse_t in enumerate(ellipses):
            plt.plot(*ellipse_t.T, label=f'Confidence ellipse {t, i}')
        plt.plot(*stochastic_trace, '.', markersize=1, label="Stochastic points {t}")

    plt.title(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$", size=15)
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20, rotation=0)
    # plt.legend()

    plt.show()


@timeit
def main():
    # show_1d_graphics(True)
    # show_2d_graphics()
    show_stochastic_2d_graphics(gamma=-0.7, sigma=0.05, epsilon=0.1, border=(-5, 5))

    show_stochastic_2d_graphics(gamma=0.7, sigma=0.04, epsilon=0.01, border=(1.5, 2))
    show_confidence_ellipse_for_equilibrium(0.7, 0.04, 0.01, (1.5, 2))

    show_stochastic_2d_graphics(0.7, 0.2, 0.01, (-3, 8))
    show_confidence_ellipses_for_k_cycle(0.7, 0.2, 0.1, (-3, 8), 0.995)

    show_deterministic_2d_graphics(-0.7, [0.3], (-5, 5))
    show_stochastic_2d_graphics(-0.7, 0.3, 0.1, (-5, 5))
    show_confidence_ellipses_for_k_cycle(-0.7, 0.3, 0.1, (-5, 5), 0.995)

    show_stochastic_2d_graphics(gamma=0.7, sigma=0.000, epsilon=0.01, border=(1.5, 2))
    show_confidence_ellipse_for_equilibrium(0.7, 0.00, 0.01, (1.5, 2))


if __name__ == '__main__':
    main()
    # make_circle_pool()
