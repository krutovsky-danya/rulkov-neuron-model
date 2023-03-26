import math
import os

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


def show_where_is_repeller(bounds, show_graphics, xs):
    ys_ = model.invert_stable_point_(xs)
    if show_graphics:
        plot.show_repeller_position(xs, ys_, bounds)


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

    xs = np.linspace(x_min, x_max, 500)
    ys = model.f(xs, gamma=gamma)

    plot.show_phase_portraits(gamma, (xs, ys), sequences, leaders)


def show_several_phase_portraits():
    for i, gamma in enumerate(np.linspace(-5, -3, 51)):
        show_portraits(gamma, *np.linspace(-2, 2, 51))

    # show_portraits(-3.3, -3.1, -0.8, 0)
    # show_portraits(-3, -3.1, -1.09, 0.4)
    # show_portraits(-3.3, -3.1, -1.09, 0.5)
    # show_portraits(-3.55, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.4875, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.494925, 0.5, steps_count=24, skip_count=650)
    # show_portraits(-3.4949, 0.5, steps_count=20, skip_count=1150)
    # show_portraits(-3.3, 0.5, steps_count=60, skip_count=10000)


def get_stable_points(x_min, x_max, count):
    xs = np.linspace(x_min, x_max, count)
    gammas = model.invert_stable_point(xs)

    return np.array((gammas, xs))


def show_stable_points():
    x_min = -4

    xs = np.linspace(x_min, 2, 500)
    gammas = model.invert_stable_point(xs)
    bounds = model.get_repeller_bounds(xs)

    plot.show_stable_points(xs, gammas, bounds)


def show_bifurcation(attractor, repeller, chaos, attracted_points, upper_attracted_points):
    stable_points_set = [attractor, repeller, chaos]

    fig, axis = plt.subplots(1, 1)
    plot.plot_bifurcation_diagram(fig, axis, attracted_points, [])
    plt.savefig("images/1d/bifurcation_only.png")
    plt.show()

    fig, axis = plt.subplots(1, 1)
    plot.plot_bifurcation_diagram(fig, axis, upper_attracted_points, stable_points_set)
    plt.savefig("images/1d/bifurcation_with_stable.png")
    plt.show()


def show_lyapunov_exponent(attractor, repeller, chaos, attracted_points):
    x_min = -4
    x_max = 2
    gammas_bound = model.invert_stable_point(np.array([x_min, x_max]))

    stable_attractor = model.get_lyapunov_exponent(attractor.T)
    stable_repeller = model.get_lyapunov_exponent(repeller.T)
    stable_chaotic = model.get_lyapunov_exponent(chaos.T)

    grouped_chaotic_points = list(model.group_chaotic_points(attracted_points))
    chaotic = model.get_lyapunov_exponent(grouped_chaotic_points)

    fig, axis = plt.subplots(1, 1)
    plot.plot_lyapunov_exponent(fig, axis, gammas_bound, chaotic, stable_attractor, stable_repeller, stable_chaotic)
    plt.savefig('images/1d/lyapunov_exponent.png')
    plt.show()


def show_1d_graphics():
    x_min = -3.9
    x_max = 1.7
    x_0 = -0.12584
    x_1 = -1.62956
    gamma_1 = -4.16187

    attractor = get_stable_points(x_min, x_1, 500)
    repeller = get_stable_points(x_1, x_0, 500)
    chaotic = get_stable_points(x_0, x_max, 500)

    bifurcation_gammas = np.linspace(gamma_1, 1, 6001)
    bifurcation_xs = np.linspace(-4, 4, 9)
    attracted_points = model.get_points_distribution(bifurcation_gammas, bifurcation_xs, 1000, 100)

    mask = attracted_points[1] > x_1
    upper_attracted_points = attracted_points[:, mask]

    show_stable_points()
    show_bifurcation(attractor, repeller, chaotic, attracted_points, upper_attracted_points)
    show_lyapunov_exponent(attractor, repeller, chaotic, upper_attracted_points)

    # show_several_phase_portraits()


def show_bifurcation_diagram_2d(gamma: float, sigmas):
    num = 7
    edges = (-5, 5)
    config = model.AttractionPoolConfiguration(gamma, sigmas.max(), edges, edges, num, 500, 100)
    _, attractors = model.get_attraction_pool(config)
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

    heatmap, attractors = model.get_attraction_pool(config)

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
    heatmap, attractors = model.get_attraction_pool(config)

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


def show_confidence_ellipses_on_attraction_pools(gamma, sigma, epsilon, border, p):
    conf = model.AttractionPoolConfiguration(gamma, sigma, border, border, density=50)
    extent = [conf.x_min, conf.x_max, conf.y_min, conf.y_max]
    heatmap, attractors = model.get_attraction_pool(conf)

    ellipses_sets = model.get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p)

    stochastic_traces = []
    for attractor in attractors:
        stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (100, 2))
        origin = np.array(list(attractor)[0])
        stochastic_trace = model.get_stochastic_coupling_trace(origin, stochastic_gammas, sigma)
        stochastic_traces.append(stochastic_trace)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(14, 7)
    fig.suptitle(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$", size=15)

    plot.plot_attraction_pool(fig, ax1, heatmap, extent)

    ax2.set_xlabel('$t$', size=15)
    ax2.set_ylabel('$x-y$', size=15)

    for stochastic_trace in stochastic_traces:
        ax1.plot(*stochastic_trace, '.')

        xs, ys = stochastic_trace
        # ax2.plot(np.abs(xs - ys))
        ax2.plot(xs - ys, '.')

    for ellipses in ellipses_sets:
        for ellipse in ellipses:
            ax1.plot(*ellipse.T)

    plt.show()


def do_something():
    gammas_count = 501
    xs_count = 501

    gamma_1 = -4.16187
    gamma_2 = -3.3
    gammas = np.linspace(gamma_1 - 0.1, gamma_2 + 0.1, gammas_count)

    x_1 = -1
    x_2 = 1
    xs = np.linspace(x_1, x_2, xs_count)
    pool = np.zeros((gammas_count, xs_count))

    for i, gamma in enumerate(gammas):
        attractors = []
        for j, x in enumerate(xs):
            xss = model.get_x_sequence(gamma, x, 100, skip_count=1000)
            xss = np.round(xss, 5)
            frozen = frozenset(list(xss))

            index = -1
            if len(frozen) < 80:
                if frozen in attractors:
                    index = attractors.index(frozen)
                else:
                    index = len(attractors)
                    attractors.append(frozen)

            index = min(3, index)

            pool[xs_count - j - 1, i] = index

    bifurcation_gammas = gammas
    bifurcation_xs = np.array([0])
    attracted_points = model.get_points_distribution(bifurcation_gammas, bifurcation_xs, 1000, 100)

    mask = attracted_points[1] > x_1
    attracted_points = attracted_points[:, mask]

    extent = [gamma_1 - 0.1, gamma_2 + 0.1, x_1, x_2]
    plt.imshow(pool, extent=extent)
    plt.plot(*attracted_points, '.r', markersize=0.01)
    plt.show()


@timeit
def main():
    show_1d_graphics()
    # show_2d_graphics()
    # show_confidence_ellipses_on_attraction_pools(gamma=-0.7, sigma=0.05, epsilon=0.1, border=(-5, 5), p=0.95)
    #
    # show_confidence_ellipses_on_attraction_pools(gamma=0.7, sigma=0.04, epsilon=0.01, border=(1.5, 2), p=0.95)
    #
    # show_confidence_ellipses_on_attraction_pools(0.7, 0.2, 0.1, (-3, 8), 0.95)
    #
    # show_confidence_ellipses_on_attraction_pools(-0.7, 0.3, 0.1, (-4, 6), 0.95)
    #
    # show_confidence_ellipses_on_attraction_pools(gamma=0.7, sigma=0.05, epsilon=0.01, border=(1.5, 2), p=0.95)

    # for epsilon in np.linspace(0.1, 0.25, 31):
    #     show_confidence_ellipses_on_attraction_pools(gamma=-0.7, sigma=0.05, epsilon=epsilon, border=(-5, 5), p=0.95)

    # for sigma in np.linspace(0, .49, 50):
    #     show_confidence_ellipses_on_attraction_pools(gamma=0.7, sigma=sigma, epsilon=0.1, border=(-6, 8), p=0.99)


if __name__ == '__main__':
    try:
        os.makedirs('images/1d')
    except OSError as error:
        print(error)

    do_something()
    show_portraits(-4, *np.linspace(-2, 2, 51))

    # main()
