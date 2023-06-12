import os

import numpy as np
import matplotlib.pyplot as plt

import model
import plot

from infrastructure import timeit


@timeit
def show_bifurcation_diagram_and_lyapunov_exponents_2d(gamma, sigmas, biff_filename, lyapunov_filename, show=True):
    num = 70
    edges = (-5, 5)
    config = model.AttractionPoolConfiguration(gamma, sigmas.max(), edges, edges, num)
    _, attractors = model.get_attraction_pool(config)

    points_sets = []
    lyapunov_origins_array = []
    for attractor in attractors:
        origin = np.array(list(attractor)[0])
        points_restarting = model.get_points_by_sigmas(origin, gamma, sigmas, steps_count=150)

        points_set = model.get_parametrized_points(sigmas, points_restarting)

        lyapunov_origins = points_restarting[:, :, -1]
        lyapunov_origins_array.append(lyapunov_origins)

        points_sets.append(points_set)

    show_bifurcation_diagram_2d(gamma, points_sets, biff_filename, show)

    show_lyapunov_exponents_2d(gamma, sigmas, lyapunov_origins_array, lyapunov_filename, show)


def show_bifurcation_diagram_2d(gamma, points_sets, filename, show):
    fig, axis = plt.subplots()

    plot.plot_bifurcation_diagram_2d(fig, axis, gamma, points_sets)

    if filename is not None:
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close()


def show_lyapunov_exponents_2d(gamma, sigmas, origins, filename, show):
    lyapunov_exponents_array = []

    np_origins = np.array(origins)
    lyapunov_exponents = model.get_lyapunov_exponents_2d(gamma, np_origins, sigmas)

    for i in range(len(origins)):
        exps = lyapunov_exponents[:, i]
        lyapunov_exponent = np.stack((sigmas, exps), axis=0)
        lyapunov_exponents_array.append(lyapunov_exponent)

    fig, axis = plt.subplots()

    plot.plot_lyapunov_exponents_2d(fig, axis, gamma, lyapunov_exponents_array)

    if filename is not None:
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close()


def get_attraction_pool_data(config: model.AttractionPoolConfiguration):
    heatmap, attractors = model.get_attraction_pool(config)
    take = max(200, config.take, config.skip)

    traces = []
    for i in range(len(attractors)):
        attractor = list(attractors[i])
        origin = np.array(attractor[0])
        trace = model.get_points(origin, config.gamma, config.sigma, take, config.skip)
        traces.append(trace)
        attractors[i] = np.unique(trace, axis=1)

    return heatmap, attractors, traces


def show_attraction_pool(config: model.AttractionPoolConfiguration, filename=None, show=True, cmap='Greens_r'):
    extent = config.get_extent()

    heatmap, attractors, traces = get_attraction_pool_data(config)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)
    plot.configure_attraction_pool_figure(fig, config.gamma, config.sigma)
    plot.plot_attraction_pool_with_attractors(fig, ax1, heatmap, extent, attractors, cmap)
    plot.plot_attraction_pool_with_attractors(fig, ax2, heatmap, extent, attractors, cmap)

    for trace in traces:
        ax2.plot(*trace)

    if filename is not None:
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close()


def show_only_pool(config, filename: str, show=True, cmap='Greens_r'):
    extent = config.get_extent()

    heatmap, attractors, traces = get_attraction_pool_data(config)

    fig, axis = plt.subplots()
    fig.set_size_inches(7, 7)

    plot.configure_attraction_pool_figure(fig, config.gamma, config.sigma)
    plot.plot_attraction_pool(fig, axis, heatmap, extent, cmap)
    for trace in traces:
        axis.plot(*trace, '.', markersize=10)

    if filename is not None:
        plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close()


def show_monostable_neuron_coupling():
    gamma = 0.7
    sigmas_for_bifurcation = np.linspace(0.48, 0, 2001)
    bif_filename = f'images/2d/bif_2d_gamma_is_{gamma}.png'
    lyapunov_filename = f'images/2d/lyapunov_gamma_is_{gamma}.png'
    show_bifurcation_diagram_and_lyapunov_exponents_2d(gamma, sigmas_for_bifurcation, bif_filename, lyapunov_filename)

    edges = (-3, 8)
    config = model.AttractionPoolConfiguration(gamma, 0.03, edges, (-3.01, 8.1), 250, take=100)
    show_only_pool(config, filename=f'images/2d/attraction_pool_single_point.png', cmap='Greens')

    config = model.AttractionPoolConfiguration(gamma, 0.2, edges, edges, 500)
    show_attraction_pool(config, f'images/2d/attraction_pool_two_cycle.png', cmap='Greens_r')


def show_bistable_neuron_coupling():
    gamma = -0.7
    sigmas_for_bifurcation = np.linspace(0.48, 0, 2001)
    bif_filename = f'images/2d/bifurcation_with_two_cycled.png'
    lyapunov_filename = f'images/2d/lyapunov_gamma_is_{gamma}.png'
    show_bifurcation_diagram_and_lyapunov_exponents_2d(gamma, sigmas_for_bifurcation, bif_filename, lyapunov_filename)

    edges = (-5, 5)

    config = model.AttractionPoolConfiguration(gamma, 0, edges, edges, 500)
    show_attraction_pool(config, filename=f'images/2d/two_two_cycles_no_interaction.png')

    config = model.AttractionPoolConfiguration(gamma, 0.05, edges, edges, 500)
    show_attraction_pool(config, filename=f"images/2d/two_two_cycles.png")

    config = model.AttractionPoolConfiguration(gamma, 0.1, edges, edges, 500)
    show_only_pool(config, filename=f'images/2d/invariant_line.png')

    config = model.AttractionPoolConfiguration(gamma, 0.3, edges, edges, 500)
    show_attraction_pool(config, filename=f'images/2d/ten_cycle.png')

    config = model.AttractionPoolConfiguration(gamma, 0.45, edges, edges, 500)
    show_attraction_pool(config, filename=f'images/2d/two_and_four_cycle.png')


def show_fractal():
    config = model.AttractionPoolConfiguration(-1.1211, 0.1, (-2, 6), (-2, 6), 1000)
    show_only_pool(config, filename=f"images/2d/fractal.png")


def show_2d_deterministic_graphics():
    try:
        os.makedirs('images/2d', exist_ok=True)
    except OSError as error:
        print(error)
    show_monostable_neuron_coupling()
    show_bistable_neuron_coupling()
    show_fractal()


if __name__ == '__main__':
    show_2d_deterministic_graphics()
