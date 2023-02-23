import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import time

import animation
import model
import plot


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


def show_stable_points(show_graphics, xs):
    gammas = model.invert_stable_point(xs)
    if show_graphics:
        plot.show_stable_points(xs, gammas)
    bounds = model.get_repeller_bounds(xs)
    print(bounds)
    if show_graphics:
        plot.show_stable_points(xs, gammas, bounds)
    return bounds


def show_where_is_repeller(bounds, show_graphics, xs):
    ys_ = model.invert_stable_point_(xs)
    if show_graphics:
        plot.show_repeller_position(xs, ys_, bounds)


def show_bifurcation(show_graphics, x_bounds, x_min):
    repeller_xs = np.linspace(*x_bounds)
    repeller_ys = model.invert_stable_point(repeller_xs)
    attractor_xs = np.linspace(x_min, min(*x_bounds))
    attractor_ys = model.invert_stable_point(attractor_xs)
    gamma_bound = model.invert_stable_point(x_bounds[1])
    chaotic_gammas = np.linspace(gamma_bound, 1, 600)
    chaotic_points = model.get_chaotic_points_cloud(chaotic_gammas)
    if show_graphics:
        plot.show_bifurcation_diagram((attractor_ys, attractor_xs), (repeller_ys, repeller_xs), chaotic_points)
    return (attractor_xs, attractor_ys), chaotic_points


def show_lyapunov(chaotic_points, attractor, show_graphics=False):
    grouped_chaotic_points = list(model.group_chaotic_points(chaotic_points))

    chaotic_lyapunov_exponent = model.get_lyapunov_exponent(grouped_chaotic_points)
    attractor_lyapunov_exponent = model.get_lyapunov_exponent(attractor)

    if show_graphics:
        plot.show_lyapunov_exponent(chaotic_lyapunov_exponent, attractor_lyapunov_exponent)


def get_sequence_generator(gamma, steps_count=100, skip_count=0):
    def func(x):
        return model.get_x_sequence(gamma, x, steps_count, skip_count)

    return func


def show_portraits(gamma, *starts, steps_count=100, skip_count=0):
    sequence_func = get_sequence_generator(gamma, steps_count, skip_count)
    sequences = list(map(sequence_func, starts))
    leaders = map(model.get_leader, sequences)

    x_min = min(map(min, sequences)) - 0.5
    x_max = max(map(max, sequences)) + 0.5

    xs = np.linspace(x_min, x_max)
    ys = model.f(xs, gamma=gamma)

    plot.show_phase_portraits(gamma, (xs, ys), sequences, leaders)


def show_several_phase_portraits(show_graphics):
    if not show_graphics:
        return
    show_portraits(-3.3, -3.1, -0.8, 0)
    show_portraits(-3, -3.1, -1.09, 0.4)
    show_portraits(-3.3, -3.1, -1.09, 0.5)
    # show_portraits(-3.55, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.4875, 0.5, steps_count=20, skip_count=150)
    # show_portraits(-3.494925, 0.5, steps_count=24, skip_count=650)
    # show_portraits(-3.4949, 0.5, steps_count=20, skip_count=1150)
    # show_portraits(-3.3, 0.5, steps_count=60, skip_count=10000)


def show_1d_graphics(show_graphics=False):
    x_min = -4

    xs = np.linspace(x_min, 1, 500)
    bounds = show_stable_points(show_graphics, xs)

    show_where_is_repeller(bounds, show_graphics, xs)

    attractor, chaotic_points = show_bifurcation(show_graphics, bounds[0], x_min)

    show_lyapunov(chaotic_points, zip(*attractor[::-1]), show_graphics)

    show_several_phase_portraits(show_graphics)


def show_bifurcation_diagram_2d(gamma: float, sigmas):
    num = 7
    points_sets = []
    for x in np.linspace(-5, 5, num):
        for y in np.linspace(-5, 5, num):
            origin = np.array((x, y))
            points_restarting = model.get_points_by_sigmas(origin, gamma, sigmas, restart=True)
            points_set = model.get_parametrized_points(sigmas, points_restarting)

            points_sets.append(points_set)

    plot.show_bifurcation_diagram_2d(gamma, points_sets)


def separate_points(last_points, cpi, steps_count):
    lined = np.reshape(last_points, (cpi * cpi * steps_count, 2))
    rounded = np.around(lined, 5)
    points, counts = np.unique(rounded, return_counts=True, axis=0)

    final_points = points.T

    co_mask = final_points[0] == final_points[1]

    return final_points[:, co_mask], final_points[:, ~co_mask]


def show_attraction_pool(config: model.AttractionPoolConfiguration, filename=None, show=True):
    gamma = config.gamma
    sigma = config.sigma

    heatmap, last_points = model.get_attraction_pool(config)

    co, anti = separate_points(last_points, config.density, config.take)

    extent = [config.x_min, config.x_max, config.y_min, config.y_max]

    plot.show_attraction_pool(gamma, sigma, heatmap, extent, co, anti, filename=filename, show=show)


def show_2d_graphics(show_graphics=False):
    gamma = -0.7
    sigmas = np.linspace(0.48, 0, 1000)

    if show_graphics:
        show_bifurcation_diagram_2d(gamma, sigmas)
        config = model.AttractionPoolConfiguration(gamma, 0.27, (-5, 5), (-5, 5), 200, 2000, 4)
        show_attraction_pool(config)


def build_attraction_pool_movie(show=True):
    s = 0.1
    filenames = []

    for i, g in enumerate(np.linspace(-1, -1.2, 201)):
        config = model.AttractionPoolConfiguration(g, s, (-2, 6), (-2, 6), 300)
        filename = f'images/image_{i}.png'
        show_attraction_pool(config, filename=filename, show=show)
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
        show_attraction_pool(config, filename=filename, show=True)
        filenames.append(filename)

    animation.build_video("animations/zooming.mov", filenames)


def make_cool_pools():
    config = model.AttractionPoolConfiguration(-1.1211, 0.1, (-10, 10), (-10, 10), 200)
    show_attraction_pool(config, filename='big_special.png', show=True)

    config.x_min, config.x_max, config.y_min, config.y_max = -1, -0.99, -1, -0.99
    show_attraction_pool(config, filename='super_small_special.png', show=True)

    config.x_min, config.x_max, config.y_min, config.y_max = -100, -1, -100, -1
    show_attraction_pool(config, filename='very_big_special.png', show=True)


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
def main():
    # show_1d_graphics(True)
    show_2d_graphics(True)


if __name__ == '__main__':
    main()
    # make_circle_pool()
