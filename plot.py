import matplotlib.pyplot as plt
from typing import Optional

import numpy as np


def show_stable_points(xs, gammas, bounds: np.ndarray, filename="images/1d/stable_points.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    ax1: plt.Axes = ax1
    ax1.set_xlabel('$x$', size=20, rotation=0)
    ax1.set_ylabel('$\\gamma$', size=20)
    ax1.set_title('$\\gamma=g(x)$')
    ax1.plot(xs, gammas)
    (x_min, x_max) = x_lim = ax1.get_xlim()
    (y_min, y_max) = y_lim = ax1.get_ylim()
    for i, bound in enumerate(bounds):
        x, gamma = bound
        ax1.plot([x, x], [gamma, y_min], '--k')
        ax1.annotate(f'$x_{i}$', (x, y_min), xytext=(x + 0.05, y_min + 0.1), size=14)

        ax1.plot([x, x_min], [gamma, gamma], '--k')
        ax1.annotate(f'$\\gamma_{i}$', (x_min + 0.05, gamma + 0.1), size=14)

    ax1.set_xlim(*x_lim)
    ax1.set_ylim(*y_lim)

    ax2.set_title('$x=g^{-1}(\\gamma)$')
    ax2.set_xlabel('$\\gamma$', size=20)
    ax2.set_ylabel('$x$', size=20, rotation=0)
    ax2.plot(gammas, xs)

    ax1.plot(*bounds.T, 'o')
    ax2.plot(*bounds.T[::-1], 'o')
    for i, bound in enumerate(bounds):
        ax1.annotate(f"$\\gamma_{i}$", bound + np.array([0, 0.2]))
        ax2.annotate(f"$\\gamma_{i}$", bound[::-1] + np.array([0.1, 0]))

    fig.suptitle("Точки покоя")
    plt.savefig(filename)
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


def show_bifurcation_diagram(attractor, repeller, chaotic_points, stable_points):
    plt.figure(figsize=(14, 7))
    plt.xlabel('$\\gamma$', size=20)
    plt.ylabel('$x$', size=20, rotation=0)

    plt.plot(*stable_points, '--')
    plt.plot(*chaotic_points, '.', markersize=0.01, label='Хаотическая')
    plt.plot(*repeller, '--', label='Неустойчивая')
    plt.plot(*attractor, label='Устойчивая')

    plt.legend()

    plt.savefig("images/1d/bifurcation.png")

    plt.show()


def plot_bifurcation_diagram(fig: plt.Figure, axis: plt.Axes, points_attraction, stable_points=None):
    fig.set_size_inches((14, 7))

    axis.set_xlabel('$\\gamma$', size=15)
    axis.set_ylabel('$x$', size=15, rotation=0)

    if stable_points is not None:
        axis.plot(*stable_points, '--k')

    axis.plot(*points_attraction, '.', markersize=0.01)

    return fig, axis


def show_lyapunov_exponent(chaotic, attractor, repeller, chaotic_but_stable):
    plt.figure(figsize=(14, 7))

    plt.plot(*chaotic, label='Неустойчивой')
    plt.plot(*attractor, label='Устойчивой')
    plt.plot(*repeller, label='Repeller')
    plt.plot(*chaotic_but_stable, label='chaotic but stable')
    plt.plot(chaotic[0, [0, -1]], np.zeros(2), 'k')

    plt.xlabel('$\\gamma$', size=20)
    plt.ylabel('$\\lambda$', size=20, rotation=0)

    plt.ylim((-4, 2))

    plt.legend()

    plt.savefig("images/1d/lyapunov_exponent.png")

    plt.show()


def show_phase_portraits(gamma, graphic, sequences, leaders):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    for xs, (x, y) in zip(sequences, leaders):
        ax1.plot(x, y)
        ax2.plot(xs)

    xs, ys = graphic
    ax1.plot(*graphic)
    ax1.plot(xs, xs)
    ax1.set_title("Iterations")
    ax1.set_xlabel("$x_t$", size=20)
    ax1.set_ylabel("$x_{t+1}$", size=20, rotation=0)

    ax2.set_title("Time series")
    ax2.set_xlabel("$t$", size=20)
    ax2.set_ylabel("$x_t$", size=20, rotation=0)

    fig.suptitle(f"Фазовый портрет $\\gamma={gamma}$")
    plt.show()


def show_bifurcation_diagram_2d(gamma: float, points_sets):
    plt.figure(figsize=(14, 7))
    for points_set in points_sets:
        plt.plot(*points_set, '.', markersize=0.1)
    plt.title(f"$\\gamma={gamma:.4f}$", size=20)
    plt.xlabel('$\\sigma$', size=20)
    plt.ylabel('$x$', size=20, rotation=0)
    plt.show()


def plot_attraction_pool(figure: plt.Figure, axis: plt.Axes, pools, extent):
    axis.set_xlabel('$x$', size=20)
    axis.set_ylabel('$y$', size=20, rotation=0)

    axis.imshow(pools.T[::-1], extent=extent)

    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])

    return figure, axis


def plot_attractors(figure: plt.Figure, axis: plt.Axes, attractors):
    for attractor in attractors:
        axis.plot(*attractor, '.')

    return figure, axis
