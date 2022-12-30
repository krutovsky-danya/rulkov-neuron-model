import numpy as np
# import matplotlib.pyplot as plt

import model
import plot


def main():
    x_min = -4

    xs = np.linspace(x_min, 1, 500)
    gammas = model.invert_stable_point(xs)
    plot.show_stable_points(xs, gammas)

    bounds = model.get_repeller_bounds(xs)

    print(bounds)

    plot.show_stable_points(xs, gammas, bounds)

    ys_ = model.invert_stable_point_(xs)

    plot.show_repeller_position(xs, ys_, bounds)

    x_bounds = bounds[0]

    repeller_xs = np.linspace(*x_bounds)
    repeller_ys = model.invert_stable_point(repeller_xs)

    attractor_xs = np.linspace(x_min, min(*x_bounds))
    attractor_ys = model.invert_stable_point(attractor_xs)

    gamma_bound = model.invert_stable_point(x_bounds[1])
    chaotic_gammas = np.linspace(gamma_bound, 1, 600)
    chaotic_points, grouped_chaotic_points = model.get_chaotic_points_cloud(chaotic_gammas)

    plot.show_bifurcation_diagram((attractor_ys, attractor_xs), (repeller_ys, repeller_xs), chaotic_points)


if __name__ == '__main__':
    main()
