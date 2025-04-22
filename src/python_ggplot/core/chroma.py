# ported code from num chroma
# todo port the unit tests too
import math
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

from python_ggplot.core.objects import Color, ColorRGBA, GGException

WHITE_X = 0.95047
WHITE_Y = 1
WHITE_Z = 1.08883


class RGBADict(TypedDict):
    r: float
    g: float
    b: float
    a: float


def c2n(hex_str: str, i: int) -> int:
    """Format int as a two digit HEX."""
    c = ord(hex_str[i])
    if ord("0") <= c <= ord("9"):
        return c - ord("0")
    elif ord("a") <= c <= ord("f"):
        return 10 + c - ord("a")
    elif ord("A") <= c <= ord("F"):
        return 10 + c - ord("A")
    else:
        raise ValueError("format is not hex")


def parse_hex(hex_str: str) -> RGBADict:
    """
    Parses colors like:
    * FF0000 -> red
    * 0000FF -> blue
    * FFFFFF -> white

    Returns:
        tuple of (r, g, b, a) values normalized between 0 and 1
    """
    if hex_str[0] == "#":
        hex_str = hex_str[1:]

    assert len(hex_str) == 6
    r = float(c2n(hex_str, 0) * 16 + c2n(hex_str, 1)) / 255
    g = float(c2n(hex_str, 2) * 16 + c2n(hex_str, 3)) / 255
    b = float(c2n(hex_str, 4) * 16 + c2n(hex_str, 5)) / 255
    return {"r": r, "g": g, "b": b, "a": 1.0}


def fixup_color(
    r: Union[int, float], g: Union[int, float], b: Union[int, float]
) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
    # port from nim chroma
    # https://github.com/treeform/chroma/blob/master/src/chroma/transformations.nim

    def fix_c(c: Union[int, float]):
        if c < 0:
            c = 0
        if isinstance(c, int):
            if c > 255:
                c = 255
        else:
            if c > 1.0:
                c = 1.0
        return c

    r = fix_c(r)
    g = fix_c(g)
    b = fix_c(b)

    return r, g, b


def hcl_to_rgb_via_luv_and_xyz(h: float, c: float, l: float):

    def hcl_to_luv(h: float, c: float, l: float):
        h_rad = math.radians(h)
        u = c * math.cos(h_rad)
        v = c * math.sin(h_rad)
        return l, u, v

    def luv_to_xyz(l: float, u: float, v: float):
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        un = 4 * xn / (xn + 15 * yn + 3 * zn)
        vn = 9 * yn / (xn + 15 * yn + 3 * zn)

        if l == 0:
            return 0, 0, 0

        y = (l + 16) / 116
        y = y**3 if y > 0.008856 else l / 903.3

        u_prime = u / (13 * l) + un if l != 0 else 0
        v_prime = v / (13 * l) + vn if l != 0 else 0

        x = 9 * u_prime * y / (4 * v_prime)
        z = (12 - 3 * u_prime - 20 * v_prime) * y / (4 * v_prime)

        return x, y, z

    def xyz_to_rgb(x: float, y: float, z: float):
        r = (3.240479 * x - 1.53715 * y - 0.498535 * z) * WHITE_Y
        g = (-0.969256 * x + 1.875992 * y + 0.041556 * z) * WHITE_Y
        b = (0.055648 * x - 0.204043 * y + 1.057311 * z) * WHITE_Y

        def gamma_correct(u: float):
            GAMMA = 2.4
            if u > 0.00304:
                return 1.055 * (u ** (1.0 / GAMMA)) - 0.055
            else:
                return 12.92 * u

        r = max(0, min(1, gamma_correct(r)))
        g = max(0, min(1, gamma_correct(g)))
        b = max(0, min(1, gamma_correct(b)))

        return (float(round(r, 5)), float(round(g, 5)), float(round(b, 5)))

    l, u, v = hcl_to_luv(h, c, l)
    x, y, z = luv_to_xyz(l, u, v)
    r, g, b = xyz_to_rgb(x, y, z)

    return {"r": r, "g": g, "b": b, "a": 1.0}


def hcl_to_rgb(h: float, c: float, l: float):
    h_rad = math.radians(h)
    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)

    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    def f_inv(t: float):
        if t > 6 / 29:
            return t**3
        else:
            return 3 * (6 / 29) ** 2 * (t - 4 / 29)

    x = f_inv(x) * 0.95047
    y = f_inv(y) * 1.0
    z = f_inv(z) * 1.08883

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def gamma_correction(c: float):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * c ** (1 / 2.4) - 0.055

    r = gamma_correction(r)
    g = gamma_correction(g)
    b = gamma_correction(b)

    r, g, b = fixup_color(r, g, b)

    return {"r": r, "g": g, "b": b, "a": 1.0}


def color_from_hsl(h: float, s: float, l: float) -> RGBADict:
    # port from nim chroma
    # https://github.com/treeform/chroma/blob/master/src/chroma/transformations.nim
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0

    if s == 0.0:
        rgb = [l, l, l]
    else:
        if l < 0.5:
            t2 = l * (1 + s)
        else:
            t2 = l + s - l * s
        t1 = 2 * l - t2

        rgb: List[float] = []
        for i in range(3):
            t3 = h + (1.0 / 3.0) * -(i - 1.0)
            if t3 < 0:
                t3 += 1
            elif t3 > 1:
                t3 -= 1

            if 6 * t3 < 1:
                val = t1 + (t2 - t1) * 6 * t3
            elif 2 * t3 < 1:
                val = t2
            elif 3 * t3 < 2:
                val = t1 + (t2 - t1) * (2 / 3 - t3) * 6
            else:
                val = t1

            rgb.append(val)

    r, g, b = rgb[0], rgb[1], rgb[2]
    a = 1.0
    r, g, b = fixup_color(r, g, b)

    return {"r": r, "g": g, "b": b, "a": a}


def to_rgba(c: int) -> tuple[int, int, int, int]:
    # todo decide what stays in color maps and what goes in chroma
    return (
        (c >> 16) & 0xFF,  # red
        (c >> 8) & 0xFF,  # green
        c & 0xFF,  # blue
        (c >> 24) & 0xFF,  # alpha
    )


def int_to_color(c: int) -> ColorRGBA:
    r, g, b, a = to_rgba(c)
    return ColorRGBA(r=r, g=g, b=b, a=a)


def parse_html_color(c: str):
    # port from chroma
    raise GGException("implement this")


def value_to_color(c: int | str) -> ColorRGBA:
    # TODO this is mean to use Value class
    # port bit by bit
    if isinstance(c, int):
        return int_to_color(c)
    elif isinstance(c, str):
        return parse_html_color(c)
    raise GGException("expected str or int")


def to_opt_color(x: Union[Color, int, str, None]) -> Optional[Color]:
    """
    TODO fix types here fine for now
    """
    color_handlers: Dict[Any, Callable[..., Color]] = {
        NoneType: lambda _: None,  # type: ignore
        Color: lambda c: c,  # type: ignore
        int: lambda c: int_to_color(c),  # type: ignore
        # Color.from_html(c) if is_valid_html_color(c) else None, # type: ignore
        # from html not ported yet
        str: lambda c: c,  # type: ignore
    }

    handler = color_handlers.get(x.__class__)  # type: ignore
    if handler is None:
        raise ValueError(f"Invalid color type: {type(x)}")

    return handler(x)  # type: ignore
