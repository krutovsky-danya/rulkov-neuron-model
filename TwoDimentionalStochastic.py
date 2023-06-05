import os

import numpy as np
from matplotlib import pyplot as plt

import model
import plot


def get_stochastic_result_from_attractors(config, attractors, heatmap):
    gamma = config.gamma
    sigma = config.sigma
    epsilon = config.epsilon
    p = config.p
    ellipses = model.get_confidence_ellipses_for_attractors(attractors, gamma, sigma, epsilon, p)

    traces = []
    for attractor in attractors:
        stochastic_gammas = gamma + epsilon * np.random.normal(0, 1, (config.stochastic_count, 2))
        origin = np.array(list(attractor)[0])
        stochastic_trace = model.get_stochastic_coupling_trace(origin, stochastic_gammas, sigma)
        traces.append(stochastic_trace)

    synchronization = []
    for stochastic_trace in traces:
        zs = model.get_synchronization_indicator(stochastic_trace)
        synchronization.append(zs)

    stochastic_result = model.StochasticResult(config, heatmap, attractors, traces, ellipses, synchronization)

    return stochastic_result


def get_stochastic_result(config: model.StochasticAttractionPoolConfiguration):
    shift = config.shift
    heatmap, attractors = model.get_attraction_pool(config, *shift)
    stochastic_result = get_stochastic_result_from_attractors(config, attractors, heatmap)

    return stochastic_result


def get_title_for_ellipses_graphic(config: model.StochasticAttractionPoolConfiguration):
    gamma = config.gamma
    sigma = config.sigma
    epsilon = config.epsilon
    p = config.p
    title = f"$\\gamma={gamma:.4f}; \\sigma={sigma:.3f};$\n$\\epsilon={epsilon:.3f}; p={p:.3f}$"
    return title


def show_stochastic_traces_on_stochastic_result(config, stochastic_result, filename, cmap='Greens_r'):
    extent = config.get_extent()
    heatmap = stochastic_result.heatmap
    stochastic = stochastic_result.traces
    ellipses = stochastic_result.ellipses

    title = get_title_for_ellipses_graphic(config)

    fig, ax1 = plt.subplots(1, 1)

    fig.set_size_inches(7, 7)

    plot.plot_stochastic_traces_on_pool(fig, ax1, title, heatmap, extent, stochastic, ellipses, cmap)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_only_stochastic_traces(config: model.StochasticAttractionPoolConfiguration, filename, cmap='Greens_r'):
    stochastic_result = get_stochastic_result(config)
    show_stochastic_traces_on_stochastic_result(config, stochastic_result, filename, cmap)


def plot_pools_clouds_and_synchronization(config, stochastic_result: model.StochasticResult, filename, cmap='Greens_r'):
    title = get_title_for_ellipses_graphic(config)
    extent = config.get_extent()

    heatmap = stochastic_result.heatmap
    stochastic_traces = stochastic_result.traces
    ellipses = stochastic_result.ellipses
    synchronization_indicators = stochastic_result.synchronization_indicators

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(14, 7)

    plot.plot_stochastic_traces_on_pool(fig, ax1, title, heatmap, extent, stochastic_traces, ellipses, cmap)
    plot.plot_synchronization_indicator(ax2, synchronization_indicators)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_confidence_ellipses(config: model.StochasticAttractionPoolConfiguration, filename, cmap='Greens_r'):
    stochastic_result = get_stochastic_result(config)
    plot_pools_clouds_and_synchronization(config, stochastic_result, filename, cmap)


def show_ellipses_on_made_pool(heatmap, attractors, config, filename):
    stochastic_result = get_stochastic_result_from_attractors(config, attractors, heatmap)
    plot_pools_clouds_and_synchronization(config, stochastic_result, filename)


def show_2d_stochastic_equilibrium_graphics():
    gamma = 0.7
    sigma = 0.01
    border = (1.7, 1.775)
    density = 50

    epsilon = 0.001
    p = 0.95

    folder = 'images/stochastic/'

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    show_only_stochastic_traces(stochastic_config, filename=f'{folder}/single_point_small_sigma.png', cmap='Greens')

    stochastic_config.sigma = 0.03
    show_only_stochastic_traces(stochastic_config, filename=f'{folder}/single_point_bigger_sigma.png', cmap='Greens')


def show_2d_stochastic_two_cycle_graphics():
    gamma = 0.7
    sigma = 0.2
    border = (-2, 7)
    density = 500

    epsilon = 0.1
    p = 0.95

    folder = 'images/stochastic/'

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=50)
    stochastic_config.shift = np.zeros(2)
    show_only_stochastic_traces(stochastic_config, filename=f'{folder}/two_cycle.png')

    density = 50
    config = model.AttractionPoolConfiguration(gamma, sigma, (-2, 2), (4, 8), density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    show_confidence_ellipses(stochastic_config, f'{folder}/two_cycle_zoomed.png', cmap='Greens')


def show_2d_stochastic_two_two_cycles_co_exist_graphics():
    gamma = -0.7
    sigma = 0.05
    border = (-5, 5)
    density = 500

    epsilon = 0.05
    p = 0.95

    folder = 'images/stochastic/'

    config = model.AttractionPoolConfiguration(gamma, 0, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    show_confidence_ellipses(stochastic_config, f'{folder}/two_2_cycles_no_interaction.png')

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=2500)

    shift = stochastic_config.shift
    heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)

    show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, f'{folder}/two_cycles_co_exists.png')

    series_folder = f'{folder}/series/two_cycles_co_exists'
    os.makedirs(series_folder, exist_ok=True)
    # for i, epsilon in enumerate(np.linspace(0.1, 0.24, 15)):
    #     for j in range(5):
    #         stochastic_config.epsilon = epsilon
    #         filename = f'{series_folder}/image_{i:04d}_{j:04d}.png'
    #         show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, filename)


def show_2d_stochastic_invariant_curve_graphics():
    gamma = -0.7
    sigma = 0.1
    border = (-2, 5)
    density = 500

    epsilon = 0.01
    p = 0.95

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    show_only_stochastic_traces(stochastic_config, filename='images/stochastic/invariant_curve.png')


def show_2d_stochastic_five_cycle_graphics():
    gamma = -0.7
    sigma = 0.3
    border = (-4, 6)
    density = 500

    epsilon = 0.1
    p = 0.95

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    stochastic_result = get_stochastic_result(stochastic_config)
    stochastic_result.traces = []
    filename = 'images/stochastic/five_cycle_with_ellipses.png'
    show_stochastic_traces_on_stochastic_result(stochastic_config, stochastic_result, filename)


def show_2d_stochastic_2_and_4_cycles_graphics():
    gamma = -0.7
    sigma = 0.45
    border = (-5, 5)
    density = 500

    epsilon = 0.03
    p = 0.95

    config = model.AttractionPoolConfiguration(gamma, sigma, border, border, density)
    stochastic_config = model.StochasticAttractionPoolConfiguration(config, epsilon, p, stochastic_count=500)
    # shift = stochastic_config.shift
    # heatmap, attractors = model.get_attraction_pool(stochastic_config, *shift)

    source_folder = 'images/stochastic/series/two_and_four'
    os.makedirs(source_folder, exist_ok=True)
    # for i, epsilon in enumerate(np.linspace(0.09, 0.095, 21)):
    #     stochastic_config.epsilon = epsilon
    #     filename = f'{source_folder}/image_{i}.png'
    #     show_ellipses_on_made_pool(heatmap, attractors, stochastic_config, filename)

    show_confidence_ellipses(stochastic_config, filename='images/stochastic/two_and_four_cycles.png')


def show_2d_stochastic_graphics():
    show_2d_stochastic_equilibrium_graphics()
    show_2d_stochastic_two_cycle_graphics()
    show_2d_stochastic_two_two_cycles_co_exist_graphics()
    show_2d_stochastic_invariant_curve_graphics()
    show_2d_stochastic_five_cycle_graphics()


def main():
    try:
        os.makedirs('images/stochastic', exist_ok=True)
    except OSError as error:
        print(error)

    show_2d_stochastic_equilibrium_graphics()
    # show_2d_stochastic_graphics()


if __name__ == '__main__':
    main()
