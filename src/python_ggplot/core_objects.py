from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, List, Literal, Optional, TypeVar, Union

from python_ggplot.cairo_backend import CairoBackend
from python_ggplot.common import linspace


class Duration:
    """
    Port from nim lang
    see here https://github.com/nim-lang/Nim/blob/version-2-2/lib/pure/times.nim#L662
    """

    def __init__(self, seconds: int, nanoseconds: int = 0):
        extra_seconds, normalized_nanoseconds = divmod(nanoseconds, 1_000_000_000)
        self.total_seconds = seconds + extra_seconds
        self.nanoseconds = normalized_nanoseconds

    def in_seconds(self) -> int:
        return self.total_seconds

    def in_milliseconds(self) -> int:
        return (self.total_seconds * 1_000) + (self.nanoseconds // 1_000_000)


def init_duration(
    nanoseconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    seconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    days: int = 0,
    weeks: int = 0,
) -> Duration:
    # TODO this needs many unit tests
    total_seconds = (
        weeks * 7 * 24 * 60 * 60
        + days * 24 * 60 * 60
        + hours * 60 * 60
        + minutes * 60
        + seconds
        + milliseconds // 1_000
        + microseconds // 1_000_000
        + nanoseconds // 1_000_000_000
    )

    total_nanoseconds = (
        (milliseconds % 1_000) * 1_000_000
        + (microseconds % 1_000_000) * 1_000
        + (nanoseconds % 1_000_000_000)
    )

    return Duration(total_seconds, total_nanoseconds)


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


class UnitType(Enum):
    POINT = auto()
    CENTIMETER = auto()
    INCH = auto()
    RELATIVE = auto()
    DATA = auto()
    STR_WIDTH = auto()
    STR_HEIGHT = auto()
    ABSTRACT = auto()

    def is_length(self):
        return self.value in (
            UnitType.POINT.value,
            UnitType.CENTIMETER.value,
            UnitType.INCH.value,
        )

    def is_absolute(self):
        return self.value in (
            UnitType.POINT.value,
            UnitType.CENTIMETER.value,
            UnitType.INCH.value,
            UnitType.STR_HEIGHT.value,
            UnitType.STR_WIDTH.value,
        )


TextType = Union[Literal[UnitType.STR_HEIGHT], Literal[UnitType.STR_WIDTH]]
LengthType = Union[
    Literal[UnitType.POINT],
    Literal[UnitType.CENTIMETER],
    Literal[UnitType.INCH],
]
