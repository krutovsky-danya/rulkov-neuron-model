import math
import os

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import time

import animation
import model
import plot

from OneDimensional import show_1d_graphics
from TwoDimensionalDeterministic import show_2d_deterministic_graphics, show_attraction_pool, \
    show_bifurcation_diagram_2d
from TwoDimentionalStochastic import show_2d_stochastic_graphics


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def build_attraction_pool_movie(show=True):
    s = 0.1
    filenames = []

    for i, g in enumerate(np.linspace(-1, -1.2, 201)):
        config = model.AttractionPoolConfiguration(g, s, (-2, 6), (-2, 6), 50)
        filename = f'animations/sources/attraction_pool/image_{i}.png'
        show_attraction_pool(config, filename=filename)
        filenames.append(filename)

    animation.build_video("animations/from_gamma.mov", filenames)


def make_cool_zooming_movie():
    s = 0.1
    g = -1.1211
    filenames = []

    frames = 50
    b1s = -np.logspace(np.log(1), np.log(0.9775), frames, base=np.e)
    b2s = np.logspace(np.log(1 + 1), np.log(-0.9776 + 1), frames, base=np.e) - 1

    for i, (b1, b2) in enumerate(zip(b1s, b2s)):
        config = model.AttractionPoolConfiguration(g, s, (b1, b2), (b1, b2), 300)
        filename = f'images/image_zoom_{i}.png'
        show_attraction_pool(config, filename=filename)
        filenames.append(filename)

    animation.build_video("animations/zooming.mov", filenames)


def make_cool_pools():
    config = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), 200)
    show_attraction_pool(config, filename='big_special.png')

    config.x_min, config.x_max, config.y_min, config.y_max = -1, -0.99, -1, -0.99
    show_attraction_pool(config, filename='super_small_special.png')

    config.x_min, config.x_max, config.y_min, config.y_max = -100, -1, -100, -1
    show_attraction_pool(config, filename='very_big_special.png')


def make_circle_pool():
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-2.5, 5), (-2.5, 5), density=100)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (-1, 4), (-1, 4), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 3.5), (-1, 1), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-1, 1), density=300)
    conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1, 2), (-0.5, 0.5), density=300)
    # conf = model.AttractionPoolConfiguration(-1.1211, 0.1, (1.6, 2), (-0.2, 0.2), density=500)
    show_attraction_pool(conf)


def make_bif_diagram_movie(source_folder, gammas):
    filenames = []
    # gammas = np.linspace(-5, 1, 501)
    sigmas_for_bifurcation = np.linspace(0.48, 0, 2001)
    for i, gamma in enumerate(gammas):
        filename = f'animations/sources/{source_folder}/diagram_{i:04d}.png'
        filenames.append(filename)
        show_bifurcation_diagram_2d(gamma, sigmas_for_bifurcation, filename)

    animation.build_video(f"animations/{source_folder}.mov", filenames)


def make_pool_on_sigmas_movie(config: model.AttractionPoolConfiguration, sigmas):
    filenames = []
    for i, sigma in enumerate(sigmas):
        config.sigma = sigma
        filename = f'animations/sources/pool_moving_sigma/pool_{i:04d}.png'
        filenames.append(filename)
        show_attraction_pool(config, filename=filename)

    animation.build_video(f"animations/pool_moving_sigma_gamma_is_{config.gamma}.mov", filenames)


@timeit
def main():
    if __name__ != '__main__':
        show_1d_graphics()
        show_2d_deterministic_graphics()
        show_2d_stochastic_graphics()


if __name__ == '__main__':
    # main()

    try:
        os.makedirs('animations/sources', exist_ok=True)
        os.makedirs('animations/sources/bif_diagrams', exist_ok=True)
        os.makedirs('animations/sources/pool_moving_sigma', exist_ok=True)
    except OSError as error:
        print(error)

    build_attraction_pool_movie()
    # gammas = np.linspace(-5, 1, 501)
    # make_bif_diagram_movie('bif_diagrams', gammas)
    # gammas = np.linspace(-3.46, -3.45, 201)  # gamma = -3.452 is interesting
    # make_bif_diagram_movie('interesting_bif_diagrams', gammas)
    # config = model.AttractionPoolConfiguration(0.7, 0, (-3, 6), (-3, 6), 250)
    # sigmas = np.linspace(0, 0.48, 49)
    # make_pool_on_sigmas_movie(config, sigmas)
    # config.gamma = -0.7
    # make_pool_on_sigmas_movie(config, sigmas)
