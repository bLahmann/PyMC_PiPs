import sys

import numpy as np
import functools
from utils import print_loading_message
from threading import Thread


_VERIFICATION_NODES = 1000


def verify_boundaries(boundaries):

    for i in range(1, len(boundaries)):

        inner_radius = functools.partial(evaluate_legendre_modes, boundaries[i - 1])
        outer_radius = functools.partial(evaluate_legendre_modes, boundaries[i])
        length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)
        print("Testing plasma " + str(i))

        names = ("inner boundary", "outer boundary", "plasma length")
        funcs = (inner_radius, outer_radius, length)

        test_thetas = np.linspace(0, np.pi, _VERIFICATION_NODES)
        test_phis = np.linspace(0, 2 * np.pi, _VERIFICATION_NODES)

        for name, func in zip(names, funcs):
            stop = False
            thread = Thread(target=print_loading_message, args=("Checking the " + name, lambda: stop))
            thread.start()
            for phi in test_phis:
                phi = np.array([phi])
                if (func(test_thetas, phi) < 0).any():
                    stop = True
                    thread.join()
                    quit("Test failed, negative " + name + " detected. Exiting now.")
            stop = True
            thread.join()


def get_scale_lengths(boundaries):

    scale_lengths = []
    for i in range(1, len(boundaries)):

        scale_lengths.append(sys.float_info.max)

        inner_radius = functools.partial(evaluate_legendre_modes, boundaries[i - 1])
        outer_radius = functools.partial(evaluate_legendre_modes, boundaries[i])
        length = lambda theta, phi: outer_radius(theta, phi) - inner_radius(theta, phi)

        test_thetas = np.linspace(0, np.pi, _VERIFICATION_NODES)
        test_phis = np.linspace(0, 2 * np.pi, _VERIFICATION_NODES)

        for phi in test_phis:
            phi = np.array([phi])
            scale_lengths[-1] = min(scale_lengths[-1], np.min(length(test_thetas, phi)))

    return scale_lengths


def evaluate_legendre_modes(modes, theta, phi):

    if theta.size > phi.size:
        result = np.zeros(theta.shape)
    else:
        result = np.zeros(phi.shape)

    for mode in modes:
        m = mode[0]
        ell = mode[1]
        magnitude = mode[2]

        norm = np.sqrt((2.0*ell + 1) * np.math.factorial(ell - m) / np.math.factorial(ell + m) / (4 * np.pi))
        result += magnitude * norm * associated_legendre_poly(theta, ell, m) * np.cos(m * phi)

    return result


def associated_legendre_poly(theta, ell, m):
    x = np.cos(theta)
    p0 = np.ones(theta.shape)
    current_ell = 0
    current_m = 0

    # Recurse until we find P(x, m, m)
    while current_m < m:
        p0 *= -(2*current_ell + 1) * np.sin(theta)
        current_m += 1
        current_ell += 1

    # If that's the one we need, return it
    if current_ell == ell:
        return p0

    # Else calculate recurse until we have P(x, m+1, m)
    p1 = p0 * x * (2*current_ell + 1)
    current_ell += 1

    # Recurse until we fine P(x, el=L, m=M)
    while current_ell < ell:
        temp = p1

        p1 = (2*current_ell + 1) * x * p1 - (current_ell + current_m) * p0
        p1 /= (current_ell - current_m + 1)

        p0 = temp
        current_ell += 1

    return p1