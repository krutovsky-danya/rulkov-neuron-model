import os

import cv2
import imageio
import numpy as np

import model
from TwoDimensionalDeterministic import show_only_pool, show_bifurcation_diagram_and_lyapunov_exponents_2d


def build_gif(gif_name, filenames):
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


def build_video(gif_name, filenames):
    img_array = []
    size = (0, 0)
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(gif_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def make_bif_diagram_and_lyapunov_exponents_movie(name, gammas, sigmas_count):
    bif_filenames = []
    lyapunov_filenames = []

    sigmas_for_bifurcation = np.linspace(0.48, 0, sigmas_count)

    bif_movie_name = name + '_bifurcation_diagram'
    lyapunov_movie_name = name + '_lyapunov_exponents'

    bif_folder = f'animations/sources/{bif_movie_name}'
    lyapunov_folder = f'animations/sources/{lyapunov_movie_name}'

    os.makedirs(bif_folder, exist_ok=True)
    os.makedirs(lyapunov_folder, exist_ok=True)

    for i, gamma in enumerate(gammas):
        bif_filename = f'{bif_folder}/diagram_{i:04d}.png'
        bif_filenames.append(bif_filename)

        lyapunov_filename = f'{lyapunov_folder}/exponents_{i:04d}.png'
        lyapunov_filenames.append(lyapunov_filename)

        show_bifurcation_diagram_and_lyapunov_exponents_2d(gamma, sigmas_for_bifurcation, bif_filename,
                                                           lyapunov_filename, show=False)

    build_video(f"animations/{bif_movie_name}.mov", bif_filenames)
    build_video(f"animations/{lyapunov_movie_name}.mov", lyapunov_filenames)


def make_cool_zooming_movie(gamma, sigma, point, radius, ratio=0.99, frames=60):
    filenames = []
    directory_name = f'zooming_gamma_is_{gamma}_sigma_is_{sigma}_ratio_is_{ratio}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_filenames = os.listdir(source_folder)

    for i in range(frames):
        x, y = point

        x_border = (x - radius, x + radius)
        y_border = (y - radius, y + radius)
        config = model.AttractionPoolConfiguration(gamma, sigma, x_border, y_border, 800)
        filename = f'image_zoom_{i:04d}.png'
        file_path = f'{source_folder}/{filename}'

        if filename not in ready_filenames:
            show_only_pool(config, filename=file_path, show=False)

        for j in range(1):
            filenames.append(file_path)

        radius *= ratio

    build_video(f'animations/{directory_name}.mov', filenames)


def make_pool_on_sigmas_movie(config: model.AttractionPoolConfiguration, sigmas, cmap='Greens_r'):
    filenames = []
    directory_name = f'pool_moving_sigma_gamma_is_{config.gamma}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_files = os.listdir(source_folder)

    for i, sigma in enumerate(sigmas):
        config.sigma = sigma
        filename = f'pool_{i:04d}.png'
        file_path = f'{source_folder}/{filename}'
        filenames.append(file_path)

        if filename not in ready_files:
            show_only_pool(config, filename=file_path, show=False, cmap=cmap)

    movie_name = f'{directory_name}.mov'

    build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


def build_attraction_pool_moving_gamma_movie(gammas, config: model.AttractionPoolConfiguration):
    filenames = []
    directory_name = f'pool_moving_gamma_when_sigma_is_{config.sigma}'
    source_folder = f'animations/sources/{directory_name}'
    os.makedirs(source_folder, exist_ok=True)
    ready_files = os.listdir(source_folder)

    for i, gamma in enumerate(gammas):
        config.gamma = gamma
        filename = f'image_{i:04d}.png'
        file_path = f'{source_folder}/{filename}'

        if filename not in ready_files:
            show_only_pool(config, filename=file_path, show=False)

        filenames.append(file_path)

    movie_name = f'{directory_name}.mov'

    build_video(f"animations/{movie_name}", filenames)

    print(f'Complete making {movie_name}')


def make_all_animations():
    _gammas = np.linspace(-0.5, 1, 151)
    make_bif_diagram_and_lyapunov_exponents_movie('behavior', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-5, -1, 401)
    make_bif_diagram_and_lyapunov_exponents_movie('investigation', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-3.46, -3.45, 201)  # gamma = -3.452 is interesting
    make_bif_diagram_and_lyapunov_exponents_movie('interesting', _gammas, sigmas_count=2001)

    x = 0.97686
    gamma = x - 4.1 / (1 + x * x)
    center = np.array([x, x])
    make_cool_zooming_movie(gamma, 0.1, center, 1, 0.95, frames=120)

    center = np.array([0.97686, 0.97686])
    make_cool_zooming_movie(-1.1211, 0.1, center, 1, 1.05, frames=120)

    _sigmas = np.linspace(0, 0.48, 48 * 2 + 1)
    _config = model.AttractionPoolConfiguration(0.7, 0, (-4, 8), (-4, 8), 250)
    make_pool_on_sigmas_movie(_config, _sigmas)

    _gammas = np.linspace(-1, -1.2, 201)
    _config = model.AttractionPoolConfiguration(0, 0.1, (-2, 6), (-2, 6), 500)
    build_attraction_pool_moving_gamma_movie(_gammas, _config)

    _sigmas = np.linspace(0, 0.48, 48 * 4 + 1)
    _config = model.AttractionPoolConfiguration(-0.7, 0, (-3, 6), (-3, 6), 500)
    make_pool_on_sigmas_movie(_config, _sigmas)


def main():
    make_all_animations()


if __name__ == '__main__':
    main()
