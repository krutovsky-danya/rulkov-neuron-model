import os
import time
from functools import wraps

import numpy as np

import animation
import model
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
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


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


@timeit
def make_bif_diagram_movie(movie_name, gammas, sigmas_count=201):
    filenames = []
    sigmas_for_bifurcation = np.linspace(0.48, 0, sigmas_count)

    os.makedirs(f'animations/sources/{movie_name}', exist_ok=True)

    for i, gamma in enumerate(gammas):
        filename = f'animations/sources/{movie_name}/diagram_{i:04d}.png'
        filenames.append(filename)
        show_bifurcation_diagram_2d(gamma, sigmas_for_bifurcation, filename, show=False)

    animation.build_video(f"animations/{movie_name}.mov", filenames)

    print(f"Complete making {movie_name} animation")


@timeit
def build_attraction_pool_moving_gamma_movie(gammas, config: model.AttractionPoolConfiguration):
    filenames = []
    directory_name = f'pool_moving_gamma_when_sigma_is_{config.sigma}'
    movie_name = f'{directory_name}.mov'

    os.makedirs(f'animations/sources/{directory_name}', exist_ok=True)

    for i, gamma in enumerate(gammas):
        config.gamma = gamma
        filename = f'animations/sources/{directory_name}/image_{i}.png'
        show_attraction_pool(config, filename=filename, show=False)
        filenames.append(filename)

    animation.build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


@timeit
def make_pool_on_sigmas_movie(config: model.AttractionPoolConfiguration, sigmas):
    filenames = []
    directory_name = f'pool_moving_sigma_gamma_is_{config.gamma}'
    movie_name = f'{directory_name}.mov'

    os.makedirs(f'animations/sources/{directory_name}', exist_ok=True)

    for i, sigma in enumerate(sigmas):
        config.sigma = sigma
        filename = f'animations/sources/{directory_name}/pool_{i:04d}.png'
        filenames.append(filename)
        show_attraction_pool(config, filename=filename, show=False)

    animation.build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


@timeit
def main():
    if __name__ != '__main__':
        show_1d_graphics()
        show_2d_deterministic_graphics()
        show_2d_stochastic_graphics()


if __name__ == '__main__':
    # main()

    seconds = (13.4391 + 18.3284 + 16.1223) * 10 * 10 + (83.6816 + 78.5314) * 10000 / 25
    print(f'Seconds: {seconds}')
    print(f'Minutes: {seconds / 60}')
    print(f'Hours: {seconds / 60 / 60}')

    _gammas = np.linspace(-1.5, 1, 251)
    make_bif_diagram_movie('bif_diagrams', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-5, -1, 401)
    make_bif_diagram_movie('investigation_of_big_diagrams', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-3.46, -3.45, 201)  # gamma = -3.452 is interesting
    make_bif_diagram_movie('interesting_bif_diagrams', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-1, -1.2, 201)
    _config = model.AttractionPoolConfiguration(0, 0.1, (-2, 6), (-2, 6), 100)
    build_attraction_pool_moving_gamma_movie(_gammas, _config)

    _sigmas = np.linspace(0, 0.48, 48 * 2 + 1)
    _config = model.AttractionPoolConfiguration(0.7, 0, (-3, 6), (-3, 6), 50)
    make_pool_on_sigmas_movie(_config, _sigmas)

    _sigmas = np.linspace(0, 0.48, 48 * 4 + 1)
    _config = model.AttractionPoolConfiguration(-0.7, 0, (-3, 6), (-3, 6), 100)
    make_pool_on_sigmas_movie(_config, _sigmas)
