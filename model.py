def f(x: float, alpha: float = 4.1, gamma: float = -3) -> float:
    return alpha / (1 + x ** 2) + gamma


def f_(x: float, alpha: float = 4.1) -> float:
    return (2 * alpha * x) / ((1 + x ** 2) ** 2)


def invert_stable_point(x: float, alpha: float = 4.1) -> float:
    return x - alpha / (1 + x ** 2)


def invert_stable_point_(x: float, alpha: float = 4.1) -> float:
    return 1 + f_(x, alpha)
