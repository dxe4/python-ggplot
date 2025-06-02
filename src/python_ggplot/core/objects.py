import math
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import Any, Generic, List, Literal, Optional, Type, TypeVar, Union

from python_ggplot.common.objects import Freezable
from python_ggplot.core.common import linspace
from python_ggplot.graphics.cairo_backend import CairoBackend

Z = TypeVar("Z", bound="GGEnum")


class GGEnum(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):  # type: ignore
        return name.lower()

    @classmethod
    def is_possible_value(cls, value: str):
        return value in [item.value for item in cls]

    @classmethod
    def value_to_name(cls, value: str):
        # TODO low priority we could cache this if it matters
        data = {item.value: item.name for item in cls}
        return data.get(data)

    @classmethod
    def value_to_item(cls: Type[Z], value: str) -> Z:
        # TODO low priority we could cache this if it matters
        data = {item.value: item for item in cls}
        try:
            return data[value]
        except KeyError as e:
            raise GGException("enum type does not exist") from e

    @classmethod
    def eitem(cls: Type[Z], value: str) -> Z:
        # TODO this is a bad name, but very convinient
        # remove or keep?
        # named it "eitem" to be able to regex it out
        return cls.value_to_item(value)


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


class MarkerKind(GGEnum):
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


class FileTypeKind(GGEnum):
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


class LineType(GGEnum):
    NONE_TYPE = auto()
    SOLID = auto()
    DASHED = auto()
    DOTTED = auto()
    DOT_DASH = auto()
    LONG_DASH = auto()
    TWO_DASH = auto()


class ErrorBarKind(GGEnum):
    LINES = auto()
    LINEST = auto()


class TextAlignKind(GGEnum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class CFontSlant(GGEnum):
    NORMAL = auto()
    ITALIC = auto()
    OBLIQUE = auto()


# TODO Move all color logic in chroma
@dataclass
class Color(Freezable):
    r: float
    g: float
    b: float
    a: float = 1.0

    def new_color_with_alpha(self, alpha: float):
        return Color(r=self.r, g=self.g, b=self.b, a=alpha)

    def __eq__(self, o: Any) -> bool:
        return self.r == o.r and self.g == o.g and self.b == o.b and self.a == o.a

    def to_rgba(self) -> "ColorRGBA":
        return ColorRGBA(
            r=int(self.r * 255),
            b=int(self.b * 255),
            g=int(self.g * 255),
            a=self.a,
        )


@dataclass
class ColorRGBA(Freezable):
    r: int
    g: int
    b: int
    a: float = 1.0

    def to_color(self):
        return Color(
            r=self.r / 255,
            b=self.b / 255,
            g=self.g / 255,
            a=self.a,
        )

    def __eq__(self, o: Any) -> bool:
        return self.r == o.r and self.g == o.g and self.b == o.b and self.a == o.a


@dataclass
class ColorHCL:
    h: float
    c: float
    l: float

    def to_rgb(self) -> Color:
        from python_ggplot.core.chroma import hcl_to_rgb_via_luv_and_xyz

        return Color(**hcl_to_rgb_via_luv_and_xyz(self.h, self.c, self.l))

    @staticmethod
    def gg_color_hue(num: int, hue_config: Optional[HueConfig] = None) -> List["Color"]:
        if not hue_config:
            hue_config = HueConfig()
        # the colors are slighly off, but moslty fine
        hues = linspace(
            hue_config.hue_start, hue_config.hue_start + 360.0, num + 1, endpoint=True
        )
        colors = [
            ColorHCL(h=h, c=hue_config.chroma, l=hue_config.luminance).to_rgb()
            for h in hues
        ]
        return colors


@dataclass
class Font:
    family: str = "sans-serif"
    size: float = 12.0
    bold: bool = False
    slant: CFontSlant = CFontSlant.NORMAL
    color: Color = field(default_factory=lambda: Color(r=0.0, g=0.0, b=0.0, a=1.0))
    align_kind: TextAlignKind = TextAlignKind.CENTER


@dataclass
class Gradient:
    colors: List[Color]
    rotation: float = 0.0


class AxisKind(GGEnum):
    X = auto()
    Y = auto()


T = TypeVar("T", int, float)


@dataclass
class Point(Generic[T]):
    x: T
    y: T


@dataclass
class Style:
    line_width: float = 1.0
    color: Color = field(default_factory=lambda: Color(r=0.0, g=0.0, b=0.0, a=0.0))
    size: float = 0.0
    line_type: LineType = LineType.NONE_TYPE
    fill_color: Optional[Color] = field(
        default_factory=lambda: Color(r=0.0, g=0.0, b=0.0, a=0.0)
    )
    marker: Optional[MarkerKind] = MarkerKind.CIRCLE
    error_bar_kind: Optional[ErrorBarKind] = ErrorBarKind.LINES
    gradient: Optional[Gradient] = None
    font: Optional[Font] = None

    def __rich_repr__(self):
        """
        TODO make this generic?
        """
        exclude_field = "gradient"
        for field in fields(self):
            if field.name != exclude_field:
                yield field.name, getattr(self, field.name)
        # this by default would print the whole set, one item at a time
        if self.gradient is not None:
            yield f"gradient -> min: {self.gradient.colors[0]} max: {self.gradient.colors[-1]} count: {len(self.gradient.colors)}"


class CompositeKind(GGEnum):
    ERROR_BAR = auto()


class TickKind(GGEnum):
    ONE_SIDE = auto()
    BOTH_SIDES = auto()


class OptionError(Exception):
    pass


K = TypeVar("K")


def either(a: Optional[K], b: Optional[K]):
    if a is not None:
        return a
    if b is not None:
        return b
    raise OptionError("Both options are None.")


@dataclass
class Scale:
    low: float
    high: float

    def merge(self, other: "Scale") -> "Scale":
        if not (self.is_empty() and math.isclose(self.low, 0)):
            return Scale(
                low=min(self.low, other.low),
                high=max(self.high, other.high),
            )
        else:
            return other

    def is_empty(self) -> bool:
        return math.isclose(self.low, self.high)

    def __eq__(self, o) -> bool:  # type: ignore
        return math.isclose(self.low, o.low) and math.isclose(self.high, o.high)  # type: ignore

    def normalise_pos(self, val: float, reverse: bool=False) -> float:
        rel_pos = (float(val) - self.low) / (self.high - self.low)
        if reverse:
            return 1 - rel_pos
        else:
            return rel_pos


class GGException(Exception):
    pass


class UnitType(GGEnum):
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

# TODO low priority this is defined in 2 places, core.objects core.common
# do general cleanup later

GREY92_DICT = {"r": 0.92, "g": 0.92, "b": 0.92, "a": 1.0}
GREY20_DICT = {"r": 0.20, "g": 0.20, "b": 0.20, "a": 1.0}
BLACK_DICT = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
WHITE_DICT = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
TRANSPARENT_DICT = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0}
GREY92 = Color(r=0.92, g=0.92, b=0.92, a=1.0)
GREY20 = Color(r=0.20, g=0.20, b=0.20, a=1.0)
BLACK = Color(r=0.0, g=0.0, b=0.0, a=1.0)
WHITE = Color(r=1.0, g=1.0, b=1.0, a=1.0)
TRANSPARENT = Color(r=0.0, g=0.0, b=0.0, a=0.0)


@dataclass
class TexOptions:
    use_te_x: bool
    tex_template: Optional[str]
    standalone: bool
    only_tik_z: bool
    caption: Optional[str]
    label: Optional[str]
    placement: str
