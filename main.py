import numpy as np
# import matplotlib.pyplot as plt

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


def main(show_graphics=False):
    x_min = -4

    xs = np.linspace(x_min, 1, 500)
    bounds = show_stable_points(show_graphics, xs)

    show_where_is_repeller(bounds, show_graphics, xs)

    attractor, chaotic_points = show_bifurcation(show_graphics, bounds[0], x_min)

    show_lyapunov(chaotic_points, zip(*attractor), show_graphics)


if __name__ == '__main__':
    main()
