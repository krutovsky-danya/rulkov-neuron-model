import os

import numpy as np
from matplotlib import pyplot as plt

import model
import plot


class StochasticResult:
    def __init__(self, config, heatmap, attractors, traces, ellipses):
        self.config = config
        self.heatmap = heatmap
        self.attractors = attractors
        self.traces = traces
        self.ellipses = ellipses


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


def get_stochastic_result(config: model.StochasticAttractionPoolConfiguration):
    gamma = config.gamma
    sigma = config.sigma
    epsilon = config.epsilon
    p = config.p
    shift = config.shift
    heatmap, attractors = model.get_attraction_pool(config, *shift)
    ellipses_sets = model.get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p)

    stochastic_traces = []
    for attractor in attractors:
        stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (config.stochastic_count, 2))
        origin = np.array(list(attractor)[0])
        stochastic_trace = model.get_stochastic_coupling_trace(origin, stochastic_gammas, sigma)
        stochastic_traces.append(stochastic_trace)

    result = StochasticResult(config, heatmap, attractors, stochastic_traces, ellipses_sets)

    return result


def get_title_for_ellipses_graphic(config: model.StochasticAttractionPoolConfiguration):
    gamma = config.gamma
    sigma = config.sigma
    epsilon = config.epsilon
    p = config.p
    title = f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$"
    return title


def show_only_stochastic_traces(config: model.StochasticAttractionPoolConfiguration, filename):
    extent = config.get_extent()
    stochastic_result = get_stochastic_result(config)
    heatmap = stochastic_result.heatmap
    stochastic = stochastic_result.traces
    ellipses = stochastic_result.ellipses

    title = get_title_for_ellipses_graphic(config)

    fig, ax1 = plt.subplots(1, 1)

    plot.plot_stochastic_traces_on_pool(fig, ax1, title, heatmap, extent, stochastic, ellipses)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_confidence_ellipses_on_attraction_pools(config: model.StochasticAttractionPoolConfiguration, filename):
    extent = config.get_extent()
    stochastic_result = get_stochastic_result(config)
    heatmap = stochastic_result.heatmap
    stochastic_traces = stochastic_result.traces
    ellipses_sets = stochastic_result.ellipses

    title = get_title_for_ellipses_graphic(config)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot.plot_stochastic_traces_on_pool(fig, ax1, title, heatmap, extent, stochastic_traces, ellipses_sets)

    ax2: plt.Axes
    ax2.set_ylim(-2, 2)

    ax2.set_xlabel('$t$', size=20)
    ax2.set_ylabel('$z$', size=20, rotation=0)

    for stochastic_trace in stochastic_traces:
        zs = model.get_synchronization_indicator(stochastic_trace)
        ax2.plot(zs, '.')

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_2d_stochastic_graphics():
    gamma = 0.7
    sigma = 0.01
    epsilon = 0.001
    border = (1.7, 1.775)
    p = 0.95
    density = 50
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    show_only_stochastic_traces(stochastic_config, filename='images/stochastic/single_point_small_sigma.png')
    stochastic_config.sigma = 0.03
    show_only_stochastic_traces(stochastic_config, filename='images/stochastic/single_point_bigger_sigma.png')

    sigma = 0.2
    border = (-2, 7)
    epsilon = 0.1
    density = 50  # * 10
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p)
    stochastic_config.shift = np.zeros(2)
    show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/two_cycle.png')

    density = 50
    config = model.AttractionPoolConfiguration(gamma, sigma, (-2, 2), (4, 8), density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p)
    show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/two_cycle_zoomed.png')

    gamma = -0.7
    sigma = 0.05
    border = (-5, 5)
    density = 50  # * 10
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    # shift = stochastic_config.shift
    # heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    # for i, epsilon in enumerate([0.1, 0.17, 0.24] * 5):
    #     stochastic_config.epsilon = epsilon
    #     show_ellipses_on_made_pool(heatmap, attractors, stochastic_config,
    #                                f'images/stochastic/two_cycles_co_exists_{i}.png')

    stochastic_config.sigma = 0.3
    stochastic_config.epsilon = 0.03
    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/five_cycle_with_ellipses.png')

    stochastic_config.sigma = 0.45
    stochastic_config.stochastic_count = 1000
    # shift = stochastic_config.shift
    # heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    # for i, epsilon in enumerate(np.linspace(0.09, 0.095, 21)):
    #     stochastic_config.epsilon = epsilon
    #     show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, f'images/stochastic/series/a_{i}.png')

    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/two_and_four_cycles.png')

    stochastic_config.sigma = 0.1
    stochastic_config.epsilon = 0.02
    stochastic_config.x_min, stochastic_config.x_max = -2, 5
    stochastic_config.y_min, stochastic_config.y_max = -2, 5
    stochastic_config.take = 64
    show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/invariant_curve.png')
    show_stochastic_2d_graphics(-0.7, 0.1, 0.02, (-2, 5))


def main():
    try:
        os.makedirs('images/stochastic', exist_ok=True)
    except OSError as error:
        print(error)

    show_2d_stochastic_graphics()


if __name__ == '__main__':
    main()
