import math
from math import factorial
from typing import Any, no_type_check

import numpy as np
from numpy.typing import NDArray

from python_ggplot.core.objects import GGException, Point


@no_type_check
def savitzky_golay(
    y: NDArray[float], window_size: int, order: int, deriv: int = 0, rate: int = 1
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


def poly_fit(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]], degree: int
) -> NDArray[np.floating[Any]]:
    poly_coeff = np.polyfit(x, y, degree)
    result = np.polyval(poly_coeff, x)
    return result


from typing import List, Optional, Tuple, Union

import numpy as np


def bincount(x: List[int], sorted_: bool = False) -> NDArray[Any]:
    if not sorted_:
        ss = sorted(x)
    else:
        ss = list(x)

    if not ss:
        return np.array([], dtype=int)

    ss_low = max(0, ss[0])
    result = np.zeros(ss[-1] - ss_low + 1, dtype=int)

    for val in ss:
        if val < 0:
            continue
        result[val - ss_low] += 1

    return result


def histogram(
    x: NDArray[np.floating[Any]],
    bins: Union[int, str],
    range: Optional[Tuple[float, float]] = None,
    normed: bool = False,
    weights: Union[Optional[List[float]], Optional[NDArray[np.floating[Any]]]] = None,
    density: bool = False,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    TODO this is a bad idea, we should use numpy histogram here
    for now its fine, need to get the public interface working first
    this is the easiest way to keep compatibility with nim side
    """

    if len(x) == 0:
        raise GGException("Cannot compute histogram of empty array!")

    if weights is not None and len(weights) != len(x):
        raise GGException(
            "The number of weights needs to be equal to the number of elements in the input sequence!"
        )

    x_array = np.asarray(x, dtype=float)

    if range is None:
        mn, mx = float(np.min(x_array)), float(np.max(x_array))
    else:
        mn, mx = range

    if mn > mx:
        raise ValueError("Max range must be larger than min range!")
    elif mn == mx:
        mn -= 0.5
        mx += 0.5

    if isinstance(bins, str):
        raise GGException(
            "Automatic choice of number of bins based on different algorithms not implemented yet."
        )

    bin_edges = np.linspace(mn, mx, bins + 1, endpoint=True)

    mask = (x_array >= mn) & (x_array <= mx)
    x_data = x_array[mask]

    norm = bins / (mx - mn)
    x_scaled = (x_data - mn) * norm
    indices = np.floor(x_scaled).astype(int)

    indices = np.clip(indices, 0, bins - 1)

    decrement = x_data < bin_edges[indices]
    indices[decrement] -= 1
    increment = (x_data >= bin_edges[indices + 1]) & (indices != (bins - 1))
    indices[increment] += 1

    hist = np.bincount(indices, minlength=bins)

    return hist, bin_edges


def create_curve(
    x: Union[float, int],
    y: Union[float, int],
    xend: Union[float, int],
    yend: Union[float, int],
    curvature: Union[float, int] = 0.3,
) -> List[Point[float]]:
    p0 = np.array([x, y])
    p2 = np.array([xend, yend])

    direction = p2 - p0
    length = np.linalg.norm(direction)

    normal = np.array([-direction[1], direction[0]]) / (length if length != 0 else 1)

    p1 = (p0 + p2) / 2 + normal * curvature * length

    t = np.linspace(0, 1, 100)
    curve = (
        ((1 - t) ** 2)[:, None] * p0
        + 2 * (1 - t)[:, None] * t[:, None] * p1
        + (t**2)[:, None] * p2
    )
    points = [Point(x=pt[0], y=pt[1]) for pt in curve]
    return points


def create_arrow(
    curve_points: List[Point[float]],
    arrow_angle: float = 25,
    arrow_size_percent: float = 8,
):
    if len(curve_points) < 2:
        raise GGException("Need at least two points to compute arrowhead.")

    p1 = curve_points[-2]
    p2 = curve_points[-1]

    angle = math.atan2(p2.y - p1.y, p2.x - p1.x)

    angle_offset = math.radians(arrow_angle)

    left_angle = angle + angle_offset
    right_angle = angle - angle_offset

    length = arrow_size_percent / 100

    left_point = Point(
        x=p2.x - length * math.cos(left_angle), y=p2.y - length * math.sin(left_angle)
    )
    right_point = Point(
        x=p2.x - length * math.cos(right_angle), y=p2.y - length * math.sin(right_angle)
    )

    return [left_point, Point(p2.x, p2.y), right_point]
