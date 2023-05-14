import os

import numpy as np
from matplotlib import pyplot as plt

import model
import plot


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
    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/single_point_small_sigma.png')
    stochastic_config.sigma = 0.03
    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/single_point_bigger_sigma.png')

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
    density = 500
    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    shift = stochastic_config.shift
    heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    for i, epsilon in enumerate([0.1, 0.17, 0.24] * 5):
        stochastic_config.epsilon = epsilon
        show_ellipses_on_made_pool(heatmap, attractors, stochastic_config,
                                   f'images/stochastic/two_cycles_co_exists_{i}.png')

    stochastic_config.sigma = 0.3
    stochastic_config.epsilon = 0.03
    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/five_cycle_with_ellipses.png')

    stochastic_config.sigma = 0.45
    stochastic_config.stochastic_count = 1000
    shift = stochastic_config.shift
    heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)
    for i, epsilon in enumerate(np.linspace(0.09, 0.095, 21)):
        stochastic_config.epsilon = epsilon
        show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, f'images/stochastic/series/a_{i}.png')

    show_confidence_ellipses_on_attraction_pools(stochastic_config,
                                                 filename='images/stochastic/two_and_four_cycles.png')

    stochastic_config.sigma = 0.1
    stochastic_config.epsilon = 0.02
    stochastic_config.x_min, stochastic_config.x_max = -2, 5
    stochastic_config.y_min, stochastic_config.y_max = -2, 5
    stochastic_config.take = 64
    show_confidence_ellipses_on_attraction_pools(stochastic_config, filename='images/stochastic/invariant_curve.png')


def main():
    try:
        os.makedirs('images/stochastic', exist_ok=True)
    except OSError as error:
        print(error)

    show_2d_stochastic_graphics()


if __name__ == '__main__':
    main()
