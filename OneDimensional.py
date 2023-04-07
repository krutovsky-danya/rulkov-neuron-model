import numpy as np
import matplotlib.pyplot as plt

import model
import plot


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


def get_sequence_generator(gamma, steps_count=100, skip_count=0):
    def func(x):
        return model.get_x_sequence(gamma, x, steps_count, skip_count)

    return func


def show_portraits(gamma, *starts, steps_count=100, skip_count=0, filename=None):
    stable_points = model.get_stable_points_fast(gamma=gamma)

    sequence_func = get_sequence_generator(gamma, steps_count, skip_count)
    sequences = list(map(sequence_func, starts))
    leaders = map(model.get_leader, sequences)

    x_min = min(map(min, sequences)) - 0.5
    x_max = max(map(max, sequences)) + 0.5

    xs = np.linspace(x_min, x_max, 500)
    ys = model.f(xs, gamma=gamma)

    for sequence in sequences:
        print(sequence[-1])
        print(*sequence)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot.plot_phase_portraits(fig, ax1, ax2, gamma, (xs, ys), sequences, leaders)

    for stable_point in stable_points:
        if x_min < stable_point < x_max:
            ax2.plot([0, steps_count], [stable_point, stable_point], '--')

    ax1.set_xlim(x_min, x_max)
    ax2.set_ylim(x_min, x_max)
    ax2.set_ylim(x_min, x_max)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def show_several_phase_portraits():
    show_portraits(1, 0, steps_count=40, filename='images/1d/single_attraction.png')
    show_portraits(-0.1, 0.15203560313851436, 1.35, steps_count=41, filename='images/1d/two_cycle.png')
    show_portraits(-1, 2.4228133045223896, steps_count=40, filename='images/1d/four_cycles.png')
    show_portraits(-1.5, 0, skip_count=100, steps_count=100, filename='images/1d/chaos.png')
    show_portraits(-1.7471, 0, skip_count=200, steps_count=200, filename='images/1d/chaotic_burst.png')
    show_portraits(-1.8, -0.028646655857585523, steps_count=40, filename='images/1d/three_cycle.png')
    show_portraits(-3.4875, 0.0025490249007873444, steps_count=40, filename='images/1d/five_cycle.png')
    show_portraits(-3, -3.1, -1.09, 0.4, filename='images/1d/from_chaos_to_stable.png')
    show_portraits(-3.3, -3.1, -1.09, 0.5, filename='images/1d/chaos_and_stable.png')


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

    show_several_phase_portraits()


if __name__ == '__main__':
    show_1d_graphics()
