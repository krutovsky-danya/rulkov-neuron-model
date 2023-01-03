import numpy as np
import matplotlib.pyplot as plt

import model
import plot


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


def show_bifurcation_diagram_2d(gamma: float, start, sigmas):
    points_restarting = model.get_points_by_sigmas(start, gamma, sigmas, skip_count=500, steps_count=20)
    restarting = model.get_parametrized_points(sigmas, points_restarting)

    points_continuing = model.get_points_by_sigmas(np.zeros(2), gamma, sigmas, False, steps_count=20, skip_count=500)
    continuing = model.get_parametrized_points(sigmas, points_continuing)

    plot.show_bifurcation_diagram_2d(gamma, restarting, continuing)


def show_2d_graphics(show_graphics=False):
    gamma = 0.3
    start = np.array([1.2, 1])
    sigmas = np.linspace(0, 0.48, 500)

    show_bifurcation_diagram_2d(gamma, start, sigmas)


def main():
    # show_1d_graphics(False)
    show_2d_graphics(True)


if __name__ == '__main__':
    main()
