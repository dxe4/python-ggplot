from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, OrderedDict, Set, Tuple


from python_ggplot.core.objects import (
    AxisKind,
    Color,
    Duration,
    GGException,
    LineType,
    MarkerKind,
    Scale,
)
from python_ggplot.gg_types import (
    ContinuousFormat,
    DataKind,
    DateTickAlgorithmKind,
    DiscreteFormat,
    DiscreteKind,
    DiscreteType,
    FormulaNode,
    GGValue,
    SecondaryAxis,
)

if typing.TYPE_CHECKING:
    from python_ggplot.gg_types import GGStyle


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


class ScaleKind(ABC):

    @property
    @abstractmethod
    def scale_type(self):
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
    alpha: float

    def update_style(self, style: "GGStyle"):
        style.alpha = self.alpha

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
    size: float

    def update_style(self, style: "GGStyle"):
        style.size = self.size

    @property
    def scale_type(self):
        return ScaleType.SIZE


class TextScale(ScaleKind):

    @property
    def scale_type(self):
        return ScaleType.TEXT


@dataclass
class ScaleValue(ABC):

    @abstractmethod
    def update_style(self, style: "GGStyle"):
        pass

    @property
    @abstractmethod
    def scale_type(self):
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

    def update_style(self, style: "GGStyle"):
        style.marker = self.marker
        style.line_type = self.line_type

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

    def update_style(self, style: "GGStyle"):
        style.fill_color = self.color
        style.color = self.color

    @property
    def scale_type(self):
        return ScaleType.FILL_COLOR


class ColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.color = self.color

    @property
    def scale_type(self):
        return ScaleType.COLOR


class TransformedDataScaleValue(ScaleValue):
    val: Optional[GGValue] = None

    @property
    def scale_type(self):
        return ScaleType.TRANSFORMED_DATA


class LinearDataScaleValue(ScaleValue):
    val: Optional[GGValue] = None

    @property
    def scale_type(self):
        return ScaleType.LINEAR_DATA


class GGScaleDiscreteKind(DiscreteKind, ABC):

    @property
    @abstractmethod
    def discrete_type(self) -> DiscreteType:
        pass


class GGScaleDiscrete(GGScaleDiscreteKind):
    value_map: OrderedDict[GGValue, "ScaleValue"]
    label_seq: List[GGValue]
    format_discrete_label: DiscreteFormat

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


class GGScaleContinuous(GGScaleDiscreteKind):
    data_scale: Scale
    format_continuous_label: ContinuousFormat

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.CONTINUOUS

    def map_data(self, df) -> List["ScaleValue"]:
        # TODO does this need to be a param or static func is fune?
        raise GGException("todo")


@dataclass
class GGScale:
    col: FormulaNode
    name: str
    ids: Set[int]
    value_kind: GGValue
    has_discreteness: bool
    num_ticks: Optional[int]
    breaks: Optional[List[float]]
    data_kind: DataKind
    scale_kind: ScaleKind
    discrete_kind: GGScaleDiscreteKind


class ScaleFreeKind(Enum):
    FIXED = auto()
    FREE_X = auto()
    FREE_Y = auto()
    FREE = auto()
