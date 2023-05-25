import os
import time
from functools import wraps

import numpy as np

import animation
import model
from OneDimensional import show_1d_graphics
from TwoDimensionalDeterministic import show_2d_deterministic_graphics, show_attraction_pool, show_only_pool, \
    show_lyapunov_exponents_2d
from TwoDimentionalStochastic import show_2d_stochastic_graphics, show_confidence_ellipses_on_attraction_pools


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


def make_cool_zooming_movie(gamma, sigma, point, radius, ratio=0.9, frames=60):
    filenames = []
    directory_name = f'zooming_gamma_is_{gamma}_sigma_is_{sigma}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_filenames = os.listdir(source_folder)

    for i in range(frames):
        x, y = point

        x_border = (x - radius, x + radius)
        y_border = (y - radius, y + radius)
        config = model.AttractionPoolConfiguration(gamma, sigma, x_border, y_border, 250)
        filename = f'{source_folder}/image_zoom_{i:04d}.png'

        if filename not in ready_filenames:
            show_only_pool(config, filename=filename, show=False)

        filenames.append(filename)

        radius *= ratio

    animation.build_video(f'animations/{directory_name}.mov', filenames)


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
def build_attraction_pool_moving_gamma_movie(gammas, config: model.AttractionPoolConfiguration):
    filenames = []
    directory_name = f'pool_moving_gamma_when_sigma_is_{config.sigma}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_files = os.listdir(source_folder)

    for i, gamma in enumerate(gammas):
        config.gamma = gamma
        filename = f'{source_folder}/image_{i}.png'

        if ready_files not in ready_files:
            show_attraction_pool(config, filename=filename, show=False)

        filenames.append(filename)

    movie_name = f'{directory_name}.mov'

    animation.build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


@timeit
def make_pool_on_sigmas_movie(config: model.AttractionPoolConfiguration, sigmas):
    filenames = []
    directory_name = f'pool_moving_sigma_gamma_is_{config.gamma}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_files = os.listdir(source_folder)

    for i, sigma in enumerate(sigmas):
        config.sigma = sigma
        filename = f'{source_folder}/pool_{i:04d}.png'
        filenames.append(filename)

        if filename not in ready_files:
            show_attraction_pool(config, filename=filename, show=False)

    movie_name = f'{directory_name}.mov'

    animation.build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


@timeit
def main():
    show_1d_graphics()
    show_2d_deterministic_graphics()
    show_2d_stochastic_graphics()


if __name__ == '__main__':
    # main()

    center = np.zeros(2)
    make_cool_zooming_movie(-3.4562, 0.125, center, 2)

    center = np.array([0.97686, 0.97686])
    make_cool_zooming_movie(-1.1211, 0.1, center, 1, frames=120)

    _sigmas = np.linspace(0, 0.48, 48 * 4 + 1)
    _config = model.AttractionPoolConfiguration(-0.7, 0, (-3, 6), (-3, 6), 500)
    make_pool_on_sigmas_movie(_config, _sigmas)

    _gammas = np.linspace(-1, -1.2, 201)
    _config = model.AttractionPoolConfiguration(0, 0.1, (-2, 6), (-2, 6), 100)
    build_attraction_pool_moving_gamma_movie(_gammas, _config)

    _sigmas = np.linspace(0, 0.48, 48 * 2 + 1)
    _config = model.AttractionPoolConfiguration(0.7, 0, (-3, 6), (-3, 6), 50)
    make_pool_on_sigmas_movie(_config, _sigmas)
