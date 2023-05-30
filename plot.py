import matplotlib.pyplot as plt

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
        ax1.annotate(f'$x_{i}$', (x, y_min), xytext=(x + 0.05, y_min + 0.1), size=20)

        ax1.plot([x, x_min], [gamma, gamma], '--k')
        ax1.annotate(f'$\\gamma_{i}$', (x_min + 0.05, gamma + 0.1), size=20)

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


def plot_bifurcation_diagram(fig: plt.Figure, axis: plt.Axes, points_attraction):
    fig.set_size_inches((14, 7))

    axis.set_xlabel('$\\gamma$', size=20)
    axis.set_ylabel('$x$', size=20, rotation=0)

    axis.plot(*points_attraction, '.', markersize=0.01)

    axis.set_ylim(-4, 4)

    return fig, axis


def plot_lyapunov_exponent(fig: plt.Figure, axis: plt.axis, gammas_bound, *exponents):
    fig.set_size_inches((14, 7))

    axis.set_xlabel('$\\gamma$', size=20)
    axis.set_ylabel('$\\lambda$', size=20, rotation=0)

    axis.plot(gammas_bound, [0, 0], 'k')

    for exponent in exponents:
        axis.plot(*exponent)

    return fig, axis


def plot_phase_portraits(fig: plt.Figure, ax1: plt.Axes, ax2: plt.Axes, gamma, graphic, sequences, leaders):
    fig.set_size_inches(14, 7)

    for xs, (x, y) in zip(sequences, leaders):
        ax1.plot(x, y)
        ax2.plot(xs)

    xs, ys = graphic
    ax1.plot(*graphic)
    ax1.plot(xs, xs)
    ax1.set_xlabel("$x_t$", size=20)
    ax1.set_ylabel("$x_{t+1}$", size=20, rotation=0)

    ax2.set_xlabel("$t$", size=20)
    ax2.set_ylabel("$x_t$", size=20, rotation=0)

    fig.suptitle(f"$\\gamma={gamma}$")


def plot_bifurcation_diagram_2d(fig: plt.Figure, axis: plt.Axes, gamma, points_sets):
    fig.set_size_inches(14, 7)
    for point_set in points_sets:
        axis.plot(*point_set, '.', markersize=0.1)

    axis.set_title(f'$\\gamma={gamma:.4f}$', size=20)
    axis.set_xlabel('$\\sigma$', size=20)
    axis.set_ylabel('$x$', size=20, rotation=0)

    return fig, axis


def configure_attraction_pool_figure(fig: plt.Figure, gamma, sigma):
    fig.suptitle(f"$\\gamma={gamma:.4f}; \\sigma={sigma:.5f}$", size=14)
    return fig


def plot_attraction_pool(figure: plt.Figure, axis: plt.Axes, pools, extent, cmap='Greens'):
    axis.set_xlabel('$x$', size=20)
    axis.set_ylabel('$y$', size=20, rotation=0)

    axis.imshow(pools.T[::-1], extent=extent, cmap=cmap)

    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])

    return figure, axis


def plot_attractors(figure: plt.Figure, axis: plt.Axes, attractors):
    for attractor in attractors:
        axis.plot(*attractor, '.')

    return figure, axis


def plot_attraction_pool_with_attractors(fig: plt.Figure, axis: plt.Axes, heatmap, extent, attractors):
    plot_attraction_pool(fig, axis, heatmap, extent)
    plot_attractors(fig, axis, attractors)


def plot_lyapunov_exponents_2d(fig: plt.Figure, axis: plt.Axes, gamma, lyapunov_exponents_array):
    fig.set_size_inches(14, 7)
    axis.set_title(f'$\\gamma={gamma:.4f}$', size=20)
    axis.set_xlabel('$\\Lambda$', size=20)
    axis.set_ylabel('$x$', size=20, rotation=0)

    for lyapunov_exponents in lyapunov_exponents_array:
        axis.plot(*lyapunov_exponents)

    return fig, axis


def plot_stochastic_traces_on_pool(fig: plt.Figure, axes: plt.Axes, title, heatmap, extent, traces, ellipses,
                                   cmap='Greens_r'):
    fig.suptitle(title, size=14)

    plot_attraction_pool(fig, axes, heatmap, extent, cmap=cmap)

    for stochastic_trace in traces:
        axes.plot(*stochastic_trace, '.')

    for ellipses in ellipses:
        for ellipse in ellipses:
            axes.plot(*ellipse.T)


def plot_synchronization_indicator(axes: plt.Axes, synchronization_indicators):
    axes.set_ylim(-2, 2)

    axes.set_xlabel('$t$', size=20)
    axes.set_ylabel('$z$', size=20, rotation=0)

    for synchronization_indicator in synchronization_indicators:
        axes.plot(synchronization_indicator)

    return axes
