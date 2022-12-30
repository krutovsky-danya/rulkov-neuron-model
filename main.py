import numpy as np
# import matplotlib.pyplot as plt

import model
import plot


def main():
    xs = np.linspace(-4, 1, 100)
    gammas = model.invert_stable_point(xs)
    plot.show_stable_points(xs, gammas)

    bounds = model.get_repeller_bounds(xs)

    print(bounds)

    plot.show_stable_points(xs, gammas, bounds)


if __name__ == '__main__':
    main()
