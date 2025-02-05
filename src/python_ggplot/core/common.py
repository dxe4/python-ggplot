import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from rich.traceback import install

if TYPE_CHECKING:
    from python_ggplot.core.objects import Font


LOG_LEVEL = logging.DEBUG
VERBOSE_OBJECT_REPR = False

REPR_CONFIG: Dict[str, bool] = {
    "GO_RECURSIVE": True,
    "GO_STYLE": True,
}

# Constants
DPI = 72.27

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger


if LOG_LEVEL == logging.DEBUG:
    install(show_locals=True)
else:
    install(show_locals=False)


def cm_to_inch(x: float) -> float:
    return x / 2.54


def inch_to_abs(x: float) -> float:
    return x * DPI


def abs_to_inch(x: float) -> float:
    return x / DPI


def inch_to_cm(x: float) -> float:
    return x * 2.54


def linspace(
    start: float, stop: float, num: int, endpoint: Optional[bool] = True
) -> List[float]:
    """
    TODO use numpy linspace?
    """
    endpoint = endpoint if endpoint is not None else True
    result: List[float] = []
    step = start
    fnum = float(num)

    if endpoint:
        diff = (stop - start) / (fnum - 1)
    else:
        diff = (stop - start) / fnum

    if diff < 0.0:
        return result

    for _ in range(num):
        result.append(step)
        step += diff
    return result


# Utilities
def to_bytes(input_list: List[int]) -> List[int]:
    """
    Converts a list of integers to a flat list of bytes (big-endian).
    """
    return [byte for value in input_list for byte in value.to_bytes(4, byteorder="big")]


def to_cairo_font_slant(font: str) -> str:
    mapping = {"Normal": "Normal", "Italic": "Italic", "Oblique": "Oblique"}
    return mapping.get(font, "Normal")


def to_cairo_font_weight(font: "Font") -> str:
    return "Bold" if font.bold else "Normal"


def nice_number(val: float, round_: bool) -> float:
    exponent = math.floor(math.log10(val))
    frac = val / math.pow(10.0, exponent)

    if round_:
        if frac < 1.5:
            nice_frac = 1.0
        elif frac < 3.0:
            nice_frac = 2.0
        elif frac < 7.0:
            nice_frac = 5.0
        else:
            nice_frac = 10.0
    else:
        if frac <= 1.0:
            nice_frac = 1.0
        elif frac <= 2.0:
            nice_frac = 2.0
        elif frac <= 5.0:
            nice_frac = 5.0
        else:
            nice_frac = 10.0

    return nice_frac * math.pow(10.0, exponent)


def is_num(x: Any):
    if isinstance(x, (int, float)):
        return True
    dtype = getattr(x, "dtype", None)  # type: ignore
    if not dtype:
        return False
    return np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
