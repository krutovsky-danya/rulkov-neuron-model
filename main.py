import math
import os

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import time

import animation
import model
import plot

from OneDimensional import show_1d_graphics
from TwoDimensionalDeterministic import show_2d_deterministic_graphics, show_attraction_pool


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
        show_attraction_pool(config, filename=filename)
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
        show_attraction_pool(config, filename=filename)
        filenames.append(filename)

    animation.build_video("animations/zooming.mov", filenames)


def make_cool_pools():
    config = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), 200)
    show_attraction_pool(config, filename='big_special.png')

    config.x_min, config.x_max, config.y_min, config.y_max = -1, -0.99, -1, -0.99
    show_attraction_pool(config, filename='super_small_special.png')

    config.x_min, config.x_max, config.y_min, config.y_max = -100, -1, -100, -1
    show_attraction_pool(config, filename='very_big_special.png')


def make_circle_pool():
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-2.5, 5), (-2.5, 5), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-1, 4), (-1, 4), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 3.5), (-1, 1), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-1, 1), density=300)
    conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-0.5, 0.5), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1.6, 2), (-0.2, 0.2), density=500)
    show_attraction_pool(conf)


def show_confidence_ellipses_on_attraction_pools(conf: model.StochasticAttractionPoolConfiguration, filename):
    shift = conf.shift
    heatmap, attractors = model.get_attraction_pool(conf, *shift)

    show_ellipses_on_made_pool(heatmap, attractors, conf, filename)


def show_ellipses_on_made_pool(heatmap, attractors, config, filename):
    gamma = config.gamma
    sigma = config.sigma
    epsilon = config.epsilon
    p = config.p
    extent = config.get_extent()
    ellipses_sets = model.get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p)

    stochastic_traces = []
    for attractor in attractors:
        stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (config.stochastic_count, 2))
        origin = np.array(list(attractor)[0])
        stochastic_trace = model.get_stochastic_coupling_trace(origin, stochastic_gammas, sigma)
        stochastic_traces.append(stochastic_trace)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(14, 7)
    fig.suptitle(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$", size=14)

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

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_2d_stochastic_graphics():
    gamma = 0.7
    sigma = 0.01
    epsilon = 0.01
    border = (1.5, 2)
    p = 0.95
    density = 50
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p)
    # show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/single_point_small_sigma.png')
    stochastic_config.sigma = 0.03
    # show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/single_point_bigger_sigma.png')

    sigma = 0.2
    border = (-2, 7)
    epsilon = 0.1
    density = 50  # * 10
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p)
    stochastic_config.shift = np.zeros(2)
    # show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/two_cycle.png')

    density = 50
    config = model.AttractionPoolConfiguration(gamma, sigma, (-2, 2), (4, 8), density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p)
    # show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/two_cycle_zoomed.png')

    gamma = -0.7
    sigma = 0.05
    border = (-5, 5)
    density = 500
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    # shift = stochastic_config.shift
    # heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    # for i, epsilon in enumerate([0.1, 0.17, 0.24] * 5):
    #     stochastic_config.epsilon = epsilon
    #     show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, f'images/stochastic/two_cycles_co_exists_{i}.png')

    # stochastic_config.sigma = 0.3
    # stochastic_config.epsilon = 0.03
    # show_confidence_ellipses_on_attraction_pools(stochastic_config,
    #                                              filename='images/stochastic/five_cycle_with_ellipses.png')

    stochastic_config.sigma = 0.45
    stochastic_config.epsilon = 0.01
    shift = stochastic_config.shift
    heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    for i, epsilon in enumerate(np.linspace(0.1, 0.12, 11)):
        stochastic_config.epsilon = epsilon
        show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, f'images/stochastic/series/a_{i}.png')
    # show_confidence_ellipses_on_attraction_pools(stochastic_config,
    #                                              filename='images/stochastic/two_and_four_cycles.png')


@timeit
def main():
    if __name__ != '__main__':
        show_1d_graphics()
        show_2d_deterministic_graphics()

    show_2d_stochastic_graphics()

    # for sigma in np.linspace(0, .49, 50):
    #     show_confidence_ellipses_on_attraction_pools(gamma=0.7, sigma=sigma, epsilon=0.1, border=(-6, 8), p=0.99)


if __name__ == '__main__':
    try:
        os.makedirs('images/stochastic')
    except OSError as error:
        print(error)
    main()
