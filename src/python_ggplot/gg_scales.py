from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

from python_ggplot.core.objects import (
    AxisKind,
    Color,
    Duration,
    LineType,
    MarkerKind,
)
from python_ggplot.gg_types import (
    DataKind,
    DateTickAlgorithmKind,
    DiscreteKind,
    FormulaNode,
    SecondaryAxis,
    Value,
)


@dataclass
class DateScale:
    name: str
    ax_kind: AxisKind
    is_timestamp: bool
    breaks: List[float]
    format_string: Optional[str] = None
    date_spacing: Optional[Duration] = None
    date_algo: DateTickAlgorithmKind = DateTickAlgorithmKind.DTA_FILTER

    def parse_date(self, date: str):
        # todo this should return datetime
        # TODO sanity check down the line, do we allow this being dynamic?
        pass


@dataclass
class ColorScale:
    name: str
    colors: List[int]


class ScaleTransform:
    pass


class ScaleType(Enum):
    LINEAR_DATA = auto()
    TRANSFORMED_DATA = auto()
    COLOR = auto()
    FILL_COLOR = auto()
    ALPHA = auto()
    SHAPE = auto()
    SIZE = auto()
    TEXT = auto()


class ScaleKind:
    pass


@dataclass
class ColorScaleData:
    color_scale: "ColorScale"


@dataclass
class LinearAndTransformScaleData:
    axis_kind: AxisKind
    reversed: bool
    transform: ScaleTransform
    secondary_axis: Optional["SecondaryAxis"]
    date_scale: Optional["DateScale"]


@dataclass
class LinearDataScale(ScaleKind):
    transform: Optional[FormulaNode] = None  # for SecondaryAxis
    data: Optional[LinearAndTransformScaleData] = None

    @property
    def scale_type(self):
        return ScaleType.LINEAR_DATA


@dataclass
class TransformedDataScale(ScaleKind):
    data: Optional[LinearAndTransformScaleData] = None

    @property
    def scale_type(self):
        return ScaleType.TRANSFORMED_DATA

    def transform(self):
        pass

    def inverse_transform(self):
        pass


class ColorScaleKind(ScaleKind):
    data: Optional[ColorScaleData] = None

    @property
    def scale_type(self):
        return ScaleType.COLOR


class FillColorScale(ScaleKind):
    data: Optional[ColorScaleData] = None

    @property
    def scale_type(self):
        return ScaleType.FILL_COLOR


class AlphaScale(ScaleKind):
    size_range = Tuple[float, float]

    @property
    def scale_type(self):
        return ScaleType.ALPHA


class ShapeScale(ScaleKind):

    @property
    def scale_type(self):
        return ScaleType.SHAPE


class SizeScale(ScaleKind):
    # low and high
    size_range = Tuple[float, float]

    @property
    def scale_type(self):
        return ScaleType.SIZE


class TextScale(ScaleKind):

    @property
    def scale_type(self):
        return ScaleType.TEXT


@dataclass
class ScaleValue:
    pass


class TextScaleValue(ScaleValue):

    @property
    def scale_type(self):
        return ScaleType.TEXT


class SizeScaleValue(ScaleValue):
    size: Optional[float] = None

    @property
    def scale_type(self):
        return ScaleType.SIZE


class ShapeScaleValue(ScaleValue):
    marker: Optional[MarkerKind] = None
    line_type: Optional[LineType] = None

    @property
    def scale_type(self):
        return ScaleType.SHAPE


class AlphaScaleValue(ScaleValue):
    alpha: Optional[float] = None

    @property
    def scale_type(self):
        return ScaleType.ALPHA


class FillColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    @property
    def scale_type(self):
        return ScaleType.FILL_COLOR


class ColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    @property
    def scale_type(self):
        return ScaleType.COLOR


class TransformedDataScaleValue(ScaleValue):
    val: Optional[Value] = None

    @property
    def scale_type(self):
        return ScaleType.TRANSFORMED_DATA


class LinearDataScaleValue(ScaleValue):
    val: Optional[Value] = None

    @property
    def scale_type(self):
        return ScaleType.LINEAR_DATA


@dataclass
class GGScale:
    col: FormulaNode
    name: str
    ids: Set[int]
    value_kind: Value
    has_discreteness: bool
    num_ticks: Optional[int]
    breaks: Optional[List[float]]
    data_kind: DataKind
    scale_kind: ScaleKind
    discrete_kind: DiscreteKind


class ScaleFreeKind(Enum):
    FIXED = auto()
    FREE_X = auto()
    FREE_Y = auto()
    FREE = auto()
