from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, TypeVar, Generic
from python_ggplot.cairo_backend import CairoBackend
from python_ggplot.common import linspace


class MarkerKind(Enum):
    CIRCLE = auto()
    CROSS = auto()
    TRIANGLE = auto()
    RHOMBUS = auto()
    RECTANGLE = auto()
    ROTCROSS = auto()
    UPSIDEDOWN_TRIANGLE = auto()
    EMPTY_CIRCLE = auto()
    EMPTY_RECTANGLE = auto()
    EMPTY_RHOMBUS = auto()


class FileTypeKind(Enum):
    SVG = auto()
    PNG = auto()
    PDF = auto()
    VEGA = auto()
    TEX = auto()


@dataclass
class Image:
    fname: str
    width: int
    height: int
    ftype: FileTypeKind
    backend: CairoBackend


@dataclass
class HueConfig:
    hue_start: float = 15.0
    chroma: float = 100.0
    luminance: float = 65.0


class LineType(Enum):
    NONE_TYPE = auto()
    SOLID = auto()
    DASHED = auto()
    DOTTED = auto()
    DOT_DASH = auto()
    LONG_DASH = auto()
    TWO_DASH = auto()


class ErrorBarKind(Enum):
    LINES = auto()
    LINEST = auto()


class TextAlignKind(Enum):
    LEFT = auto()
    CENRTER = auto()
    RIGHT = auto()


class CFontSlant(Enum):
    NORMAL = auto()
    ITALIC = auto()
    OBLIQUE = auto()


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float


@dataclass
class ColorHCL:
    h: float
    c: float
    l: float

    @staticmethod
    def gg_color_hue(num: int, hue_config: HueConfig) -> List["ColorHCL"]:
        hues = linspace(
            hue_config.hue_start, hue_config.hue_start + 360.0, num + 1, endpoint=True
        )
        colors = [
            ColorHCL(h=h, c=hue_config.chroma, l=hue_config.luminance) for h in hues
        ]
        return colors


@dataclass
class Font:
    family: str = "sans-serif"
    size: float = 12.0
    bold: bool = False
    slant: CFontSlant = CFontSlant.NORMAL
    color: Color = Color(r=0.0, g=0.0, b=0.0, a=1.0)
    align_kind: TextAlignKind = TextAlignKind.CENRTER


@dataclass
class Gradient:
    colors: List[Color]
    rotation: float


class AxisKind(Enum):
    X = auto()
    Y = auto()


T = TypeVar("T", int, float)


@dataclass
class Point(Generic[T]):
    x: T
    y: T


@dataclass
class Style:
    color: Color
    size: float
    line_type: LineType
    line_width: float
    fill_color: Color
    marker: MarkerKind
    error_bar_kind: ErrorBarKind
    gradient: Optional[Gradient]
    font: Font


class CompositeKind(Enum):
    ERROR_BAR = auto()


class TickKind(Enum):
    ONE_SIDE = auto()
    BOTH_SIDES = auto()


class OptionError(Exception):
    pass


def either(a, b):
    if a is not None:
        return a
    if b is not None:
        return b
    raise OptionError("Both options are None.")


@dataclass
class Scale:
    low: float
    high: float

    def __eq__(self, o):
        return self.low == o.low and self.high == o.high


class GGException(Exception):
    pass
