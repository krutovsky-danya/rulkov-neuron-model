import matplotlib.pyplot as plt
import numpy as np

import model


def plot_stable_points(x_min: float = -4, x_max: float = 1, count: int = 100, bounds: bool = False) -> plt.Figure:
    xs = np.linspace(x_min, x_max, count)
    ys = model.invert_stable_point(xs)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 7)

    ax1.set_title('$x=g(\\gamma)$')
    ax1.set_xlabel('$\\gamma$', size=20)
    ax1.set_ylabel('$x$', size=20, rotation=0)
    ax1.plot(ys, xs)

    ax2.set_xlabel('$x$', size=20, rotation=0)
    ax2.set_ylabel('$\\gamma$', size=20)
    ax2.set_title('$\\gamma=g^{-1}(x)$')
    ax2.plot(xs, ys)

    fig.suptitle("Stable points")
    return fig
