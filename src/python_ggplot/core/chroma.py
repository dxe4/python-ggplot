# ported code from num chroma
# todo port the unit tests too
import math
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

from python_ggplot.core.objects import Color, ColorRGBA, GGException

WHITE_X = 0.95047
WHITE_Y = 1
WHITE_Z = 1.08883

COMMON_COLORS = {
    "transparent": (0, 0, 0, 0),
    "aliceblue": (240, 248, 255, 1),
    "antiquewhite": (250, 235, 215, 1),
    "aqua": (0, 255, 255, 1),
    "aquamarine": (127, 255, 212, 1),
    "azure": (240, 255, 255, 1),
    "beige": (245, 245, 220, 1),
    "bisque": (255, 228, 196, 1),
    "black": (0, 0, 0, 1),
    "blanchedalmond": (255, 235, 205, 1),
    "blue": (0, 0, 255, 1),
    "blueviolet": (138, 43, 226, 1),
    "brown": (165, 42, 42, 1),
    "burlywood": (222, 184, 135, 1),
    "cadetblue": (95, 158, 160, 1),
    "chartreuse": (127, 255, 0, 1),
    "chocolate": (210, 105, 30, 1),
    "coral": (255, 127, 80, 1),
    "cornflowerblue": (100, 149, 237, 1),
    "cornsilk": (255, 248, 220, 1),
    "crimson": (220, 20, 60, 1),
    "cyan": (0, 255, 255, 1),
    "darkblue": (0, 0, 139, 1),
    "darkcyan": (0, 139, 139, 1),
    "darkgoldenrod": (184, 134, 11, 1),
    "darkgray": (169, 169, 169, 1),
    "darkgreen": (0, 100, 0, 1),
    "darkkhaki": (189, 183, 107, 1),
    "darkmagenta": (139, 0, 139, 1),
    "darkolivegreen": (85, 107, 47, 1),
    "darkorange": (255, 140, 0, 1),
    "darkorchid": (153, 50, 204, 1),
    "darkred": (139, 0, 0, 1),
    "darksalmon": (233, 150, 122, 1),
    "darkseagreen": (143, 188, 143, 1),
    "darkslateblue": (72, 61, 139, 1),
    "darkslategray": (47, 79, 79, 1),
    "darkturquoise": (0, 206, 209, 1),
    "darkviolet": (148, 0, 211, 1),
    "deeppink": (255, 20, 147, 1),
    "deepskyblue": (0, 191, 255, 1),
    "dimgray": (105, 105, 105, 1),
    "dodgerblue": (30, 144, 255, 1),
    "firebrick": (178, 34, 34, 1),
    "floralwhite": (255, 250, 240, 1),
    "forestgreen": (34, 139, 34, 1),
    "fuchsia": (255, 0, 255, 1),
    "gainsboro": (220, 220, 220, 1),
    "ghostwhite": (248, 248, 255, 1),
    "gold": (255, 215, 0, 1),
    "goldenrod": (218, 165, 32, 1),
    "gray": (128, 128, 128, 1),
    "green": (0, 128, 0, 1),
    "greenyellow": (173, 255, 47, 1),
    "honeydew": (240, 255, 240, 1),
    "hotpink": (255, 105, 180, 1),
    "indianred": (205, 92, 92, 1),
    "indigo": (75, 0, 130, 1),
    "ivory": (255, 255, 240, 1),
    "khaki": (240, 230, 140, 1),
    "lavender": (230, 230, 250, 1),
    "lavenderblush": (255, 240, 245, 1),
    "lawngreen": (124, 252, 0, 1),
    "lemonchiffon": (255, 250, 205, 1),
    "lightblue": (173, 216, 230, 1),
    "lightcoral": (240, 128, 128, 1),
    "lightcyan": (224, 255, 255, 1),
    "lightgray": (211, 211, 211, 1),
    "lightgreen": (144, 238, 144, 1),
    "lightpink": (255, 182, 193, 1),
    "lightsalmon": (255, 160, 122, 1),
    "lightseagreen": (32, 178, 170, 1),
    "lightskyblue": (135, 206, 250, 1),
    "lightslategray": (119, 136, 153, 1),
    "lightsteelblue": (176, 196, 222, 1),
    "lightyellow": (255, 255, 224, 1),
    "lime": (0, 255, 0, 1),
    "limegreen": (50, 205, 50, 1),
    "linen": (250, 240, 230, 1),
    "magenta": (255, 0, 255, 1),
    "maroon": (128, 0, 0, 1),
    "mediumaquamarine": (102, 205, 170, 1),
    "mediumblue": (0, 0, 205, 1),
    "mediumorchid": (186, 85, 211, 1),
    "mediumpurple": (147, 112, 219, 1),
    "mediumseagreen": (60, 179, 113, 1),
    "mediumslateblue": (123, 104, 238, 1),
    "mediumspringgreen": (0, 250, 154, 1),
    "mediumturquoise": (72, 209, 204, 1),
    "mediumvioletred": (199, 21, 133, 1),
    "midnightblue": (25, 25, 112, 1),
    "mintcream": (245, 255, 250, 1),
    "mistyrose": (255, 228, 225, 1),
    "moccasin": (255, 228, 181, 1),
    "navajowhite": (255, 222, 173, 1),
    "navy": (0, 0, 128, 1),
    "oldlace": (253, 245, 230, 1),
    "olive": (128, 128, 0, 1),
    "olivedrab": (107, 142, 35, 1),
    "orange": (255, 165, 0, 1),
    "orangered": (255, 69, 0, 1),
    "orchid": (218, 112, 214, 1),
    "palegoldenrod": (238, 232, 170, 1),
    "palegreen": (152, 251, 152, 1),
    "paleturquoise": (175, 238, 238, 1),
    "palevioletred": (219, 112, 147, 1),
    "papayawhip": (255, 239, 213, 1),
    "peachpuff": (255, 218, 185, 1),
    "peru": (205, 133, 63, 1),
    "pink": (255, 192, 203, 1),
    "plum": (221, 160, 221, 1),
    "powderblue": (176, 224, 230, 1),
    "purple": (128, 0, 128, 1),
    "rebeccapurple": (102, 51, 153, 1),
    "red": (255, 0, 0, 1),
    "rosybrown": (188, 143, 143, 1),
    "royalblue": (65, 105, 225, 1),
    "saddlebrown": (139, 69, 19, 1),
    "salmon": (250, 128, 114, 1),
    "sandybrown": (244, 164, 96, 1),
    "seagreen": (46, 139, 87, 1),
    "seashell": (255, 245, 238, 1),
    "sienna": (160, 82, 45, 1),
    "silver": (192, 192, 192, 1),
    "skyblue": (135, 206, 235, 1),
    "slateblue": (106, 90, 205, 1),
    "slategray": (112, 128, 144, 1),
    "snow": (255, 250, 250, 1),
    "springgreen": (0, 255, 127, 1),
    "steelblue": (70, 130, 180, 1),
    "tan": (210, 180, 140, 1),
    "teal": (0, 128, 128, 1),
    "thistle": (216, 191, 216, 1),
    "tomato": (255, 99, 71, 1),
    "turquoise": (64, 224, 208, 1),
    "violet": (238, 130, 238, 1),
    "wheat": (245, 222, 179, 1),
    "white": (255, 255, 255, 1),
    "whitesmoke": (245, 245, 245, 1),
    "yellow": (255, 255, 0, 1),
    "yellowgreen": (154, 205, 50, 1),
}


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


def to_rgba(c: int) -> tuple[int, int, int, float]:
    # todo decide what stays in color maps and what goes in chroma
    return (
        (c >> 16) & 0xFF,  # red
        (c >> 8) & 0xFF,  # green
        c & 0xFF,  # blue
        ((c >> 24) & 0xFF) / 255,  # alpha
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


def str_to_color(c: str) -> Optional[Color]:
    # we need to implement hex and more, its easy just not prioritised
    color = COMMON_COLORS.get(c)
    if color is not None:
        color = ColorRGBA(color[0], color[1], color[2], color[3]).to_color()
    return color


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
        str: str_to_color,  # type: ignore
    }

    handler = color_handlers.get(x.__class__)  # type: ignore
    if handler is None:
        raise ValueError(f"Invalid color type: {type(x)}")

    return handler(x)  # type: ignore
