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
