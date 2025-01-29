from math import factorial
from typing import Any

import numpy as np
from numpy.typing import NDArray


def savitzky_golay(
    y: NDArray[Any], window_size: int, order: int, deriv: int = 0, rate: int = 1
):
    """
    this is a copy from scipy
    really dont want scipy as dependency
    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    if this doesnt work would be more keen porting from rust or c++ than having whole scipy as dependacy
    https://github.com/tpict/savgol-rs/blob/main/src/lib.rs

    seems is from an old version of numpy
    """
    try:
        window_size = np.abs(window_size)
        order = np.abs(order)
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.matrix(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)  # type: ignore
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")  # type: ignore


def poly_fit(x: NDArray[np.float64], y: NDArray[np.float64], degree: int):
    poly_coeff = np.polyfit(x, y, degree)
    result = np.polyval(poly_coeff, x)
    return result
