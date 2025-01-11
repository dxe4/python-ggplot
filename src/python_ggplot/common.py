from typing import List, Optional

# Constants
DPI = 72.27

GREY92 = {"r": 0.92, "g": 0.92, "b": 0.92, "a": 1.0}
GREY20 = {"r": 0.20, "g": 0.20, "b": 0.20, "a": 1.0}
BLACK = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
WHITE = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
TRANSPARENT = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0}


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
    endpoint = endpoint if endpoint is not None else True
    result = []
    step = start
    fnum = float(num)

    diff = (stop - start) / (fnum - 1 if endpoint else fnum)
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
