import os

import imageio
import cv2
import numpy as np

from TwoDimensionalDeterministic import show_bifurcation_diagram_2d


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


def make_bif_diagram_movie(movie_name, gammas, sigmas_count=201):
    filenames = []
    sigmas_for_bifurcation = np.linspace(0.48, 0, sigmas_count)

    os.makedirs(f'animations/sources/{movie_name}', exist_ok=True)

    for i, gamma in enumerate(gammas):
        filename = f'animations/sources/{movie_name}/diagram_{i:04d}.png'
        filenames.append(filename)
        show_bifurcation_diagram_2d(gamma, sigmas_for_bifurcation, filename, show=False)

    build_video(f"animations/{movie_name}.mov", filenames)

    print(f"Complete making {movie_name} animation")


def main():
    _gammas = np.linspace(-1.5, 1, 251)
    make_bif_diagram_movie('bif_diagrams', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-5, -1, 401)
    make_bif_diagram_movie('investigation_of_big_diagrams', _gammas, sigmas_count=2001)

    _gammas = np.linspace(-3.46, -3.45, 201)  # gamma = -3.452 is interesting
    make_bif_diagram_movie('interesting_bif_diagrams', _gammas, sigmas_count=2001)


if __name__ == '__main__':
    main()
