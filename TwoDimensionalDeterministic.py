import os

import numpy as np
import matplotlib.pyplot as plt

import model
import plot


def show_bifurcation_diagram_2d(gamma: float, sigmas, filename=None):
    num = 7
    edges = (-5, 5)
    config = model.AttractionPoolConfiguration(gamma, sigmas.max(), edges, edges, num)
    _, attractors = model.get_attraction_pool(config)
    points_sets = []

    for attractor in attractors:
        origin = np.array(list(attractor)[0])
        points_restarting = model.get_points_by_sigmas(origin, gamma, sigmas, steps_count=150)
        points_set = model.get_parametrized_points(sigmas, points_restarting)

        points_sets.append(points_set)

    fig, axis = plt.subplots()

    plot.plot_bifurcation_diagram_2d(fig, axis, gamma, points_sets)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def get_attraction_pool_data(config: model.AttractionPoolConfiguration):
    heatmap, attractors = model.get_attraction_pool(config)

    traces = []
    for i in range(len(attractors)):
        attractor = list(attractors[i])
        origin = np.array(attractor[0])
        trace = model.get_points(origin, config.gamma, config.sigma, 200)
        traces.append(trace)
        attractors[i] = np.unique(trace, axis=1)

    return heatmap, attractors, traces


def show_attraction_pool(config: model.AttractionPoolConfiguration, filename=None):
    extent = config.get_extent()

    heatmap, attractors, traces = get_attraction_pool_data(config)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot.configure_attraction_pool_figure(fig, config.gamma, config.sigma)
    plot.plot_attraction_pool_with_attractors(fig, ax1, heatmap, extent, attractors)
    plot.plot_attraction_pool_with_attractors(fig, ax2, heatmap, extent, attractors)

    for trace in traces:
        ax2.plot(*trace, '-')

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_only_pool(config, filename):
    extent = config.get_extent()

    heatmap, attractors = model.get_attraction_pool(config)

    for i in range(len(attractors)):
        listed = list(attractors[i])
        attractors[i] = np.array(listed).T

    fig, axis = plt.subplots()

    plot.configure_attraction_pool_figure(fig, config.gamma, config.sigma)
    plot.plot_attraction_pool(fig, axis, heatmap, extent)
    for attractor in attractors:
        axis.plot(*attractor, '.', markersize=10)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_monostable_neuron_coupling():
    gamma = 0.7
    sigmas_for_bifurcation = np.linspace(0.48, 0, 2001)
    show_bifurcation_diagram_2d(gamma, sigmas_for_bifurcation, f"images/2d/bif_2d_gamma_is_{gamma}.png")

    edges = (-3, 8)
    config = model.AttractionPoolConfiguration(gamma, 0.03, edges, edges, 50)
    show_only_pool(config, filename=f"images/2d/attraction_pool_single_point.png")

    config = model.AttractionPoolConfiguration(gamma, 0.2, edges, edges, 500)
    show_attraction_pool(config, f'images/2d/attraction_pool_two_cycle.png')


def show_bistable_neuron_coupling():
    gamma = -0.7
    sigmas_for_bifurcation = np.linspace(0.48, 0, 2001)
    show_bifurcation_diagram_2d(gamma, sigmas_for_bifurcation, f"images/2d/bifurcation_with_two_cycled.png")

    edges = (-5, 5)
    config = model.AttractionPoolConfiguration(gamma, 0.05, edges, edges, 500)
    show_attraction_pool(config, filename=f"images/2d/two_two_cycles.png")

    config = model.AttractionPoolConfiguration(gamma, 0.1, edges, edges, 500)
    show_attraction_pool(config, filename=f"images/2d/invariant_line.png")

    config = model.AttractionPoolConfiguration(gamma, 0.3, edges, edges, 500)
    show_attraction_pool(config, filename=f"images/2d/ten_cycle.png")

    config = model.AttractionPoolConfiguration(gamma, 0.45, edges, edges, 500)
    show_attraction_pool(config, filename=f"images/2d/two_and_four_cycle.png")


def show_2d_deterministic_graphics():
    try:
        os.makedirs('images/2d')
    except OSError as error:
        print(error)
    show_monostable_neuron_coupling()
    show_bistable_neuron_coupling()


if __name__ == '__main__':
    show_2d_deterministic_graphics()