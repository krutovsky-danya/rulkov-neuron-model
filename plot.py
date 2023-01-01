import matplotlib.pyplot as plt
from typing import Optional

import numpy as np


def show_stable_points(xs, gammas, bounds: Optional[np.ndarray] = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 7)

    ax1.set_title('$x=g(\\gamma)$')
    ax1.set_xlabel('$\\gamma$', size=20)
    ax1.set_ylabel('$x$', size=20, rotation=0)
    ax1.plot(gammas, xs)

    ax2.set_xlabel('$x$', size=20, rotation=0)
    ax2.set_ylabel('$\\gamma$', size=20)
    ax2.set_title('$\\gamma=g^{-1}(x)$')
    ax2.plot(xs, gammas)

    if bounds is not None:
        ax1.plot(*bounds[::-1], 'o')
        ax2.plot(*bounds, 'o')

    fig.suptitle("Stable points")
    plt.show()


def show_repeller_position(xs, ys, bounds):
    bx1, bx2 = bounds[0]
    mask = (bx1 < xs) & (xs < bx2)

    plt.figure(figsize=(14, 7))
    plt.ylabel("$\\gamma_x'$", size=20, rotation=0)
    plt.xlabel('$x$', size=20, rotation=0)
    plt.title("Where is repeller")
    plt.plot([xs[0], xs[-1]], [0, 0], 'k')
    plt.plot(xs, ys)
    plt.fill_between(xs[mask], ys[mask])
    plt.plot(bounds[0], [0, 0])
    plt.show()


def show_bifurcation_diagram(attractor, repeller, chaotic_points):
    plt.figure(figsize=(14, 7))
    plt.xlabel('$\\gamma$', size=20)
    plt.ylabel('$x$', size=20, rotation=0)

    plt.plot(*chaotic_points, '.', markersize=0.01, label='Хаотическая')
    plt.plot(*repeller, label='Неустойчивая')
    plt.plot(*attractor, label='Устойчивая')

    plt.legend()
    plt.show()


def show_lyapunov_exponent(chaotic, attractor):
    plt.figure(figsize=(20, 10))

    plt.plot(*chaotic, label='Неустойчивой')
    plt.plot(*attractor, label='Устойчивой')
    plt.plot(chaotic[0, [0, -1]], np.zeros(2))

    plt.xlabel('$\\gamma$', size=20)
    plt.ylabel('$\\lambda$', size=20, rotation=0)

    plt.legend()

    plt.show()
