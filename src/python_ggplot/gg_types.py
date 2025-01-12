from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, OrderedDict, Set, Tuple, Union

from python_ggplot.core_objects import (
    AxisKind,
    Color,
    Duration,
    ErrorBarKind,
    Font,
    GGException,
    LineType,
    MarkerKind,
    Scale,
)

COUNT_COL = "counts_GGPLOTNIM_INTERNAL"
PREV_VALS_COL = "prevVals_GGPLOTNIM_INTERNAL"
SMOOTH_VALS_COL = "smoothVals_GGPLOTNIM_INTERNAL"


class Value:
    pass


class VString(Value):
    data: str


class VInt(Value):
    data: int


class VFloat(Value):
    data: float


class VBool(Value):
    data: bool


class VObject(Value):
    fields: OrderedDict[str, Value]


class VNull(Value):
    pass


class ScaleTransform:
    # TODO impl
    pass


class FormulaNode:
    pass


class AestheticError(Exception):
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


class ScaleKind:

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        pass


@dataclass
class LinearDataScale(ScaleKind):
    transform: Optional[FormulaNode] = None
    data: Optional[LinearAndTransformScaleData] = None

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(kind=cls(), val=kwargs.get("val"))


@dataclass
class TransformedDataScale(ScaleKind):
    data: Optional[LinearAndTransformScaleData] = None

    def transform(self):
        pass

    def inverse_transform(self):
        pass

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(kind=cls(), val=kwargs.get("val"))


class ColorScaleKind(ScaleKind):
    data: Optional[ColorScaleData] = None

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(kind=cls(), color=kwargs.get("color"))


class FillColorScale(ScaleKind):
    data: Optional[ColorScaleData] = None

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(kind=cls(), color=kwargs.get("color"))


class AlphaScale(ScaleKind):
    size_range = Tuple[float, float]

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(kind=cls(), alpha=kwargs.get("alpha"))


class ShapeScale(ScaleKind):

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(
            kind=cls(),
            marker=kwargs.get("marker"),
            line_type=kwargs.get("line_type"),
        )


class SizeScale(ScaleKind):
    # low and high
    size_range = Tuple[float, float]

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(
            kind=cls(),
            size=kwargs.get("size"),
        )


class TextScale(ScaleKind):

    @classmethod
    def create_scale_value(cls, **kwargs) -> "ScaleValue":
        return ScaleValue(
            kind=cls(),
        )


class PositionKind(Enum):
    PK_IDENTITY = auto()
    PK_STACK = auto()
    PK_DODGE = auto()
    PK_FILL = auto()


class StatKind(Enum):
    ST_IDENTITY = auto()
    ST_COUNT = auto()
    ST_BIN = auto()
    ST_SMOOTH = auto()


@dataclass
class ScaleValue:
    kind: ScaleKind
    color: Optional[Color] = None
    alpha: Optional[float] = None
    size: Optional[float] = None
    marker: Optional[MarkerKind] = None
    line_type: Optional[LineType] = None
    val: Optional[Value] = None

    @staticmethod
    def create_from_kind(kind: ScaleKind, **kwargs) -> ScaleValue:
        return kind.create_scale_value(**kwargs)


# todo refactor
DiscreteFormat = Callable[["Value"], str]
ContinuousFormat = Callable[[float], str]


class DiscreteKind:
    pass


class Discrete(DiscreteKind):
    value_map: OrderedDict[Value, ScaleValue]
    label_seq: List[Value]
    format_discrete_label: DiscreteFormat


class Continuous(DiscreteKind):
    data_scale: Scale
    format_continuous_label: ContinuousFormat

    def map_data(self, df) -> List[ScaleValue]:
        # TODO
        raise GGException("todo")


@dataclass
class Aesthetics:
    x: Optional["ScaleValue"] = None
    x_min: Optional["ScaleValue"] = None
    x_max: Optional["ScaleValue"] = None
    y: Optional["ScaleValue"] = None
    y_min: Optional["ScaleValue"] = None
    y_max: Optional["ScaleValue"] = None
    fill: Optional["ScaleValue"] = None
    color: Optional["ScaleValue"] = None
    alpha: Optional["ScaleValue"] = None
    size: Optional["ScaleValue"] = None
    shape: Optional["ScaleValue"] = None
    width: Optional["ScaleValue"] = None
    height: Optional["ScaleValue"] = None
    text: Optional["ScaleValue"] = None
    y_ridges: Optional["ScaleValue"] = None
    weight: Optional["ScaleValue"] = None
    scale_kind: ScaleKind
    position_kind: PositionKind
    stat_kind: StatKind
    discrete_kind: DiscreteKind


@dataclass
class SecondaryAxis:
    name: str
    ax_kind: AxisKind
    sc_kind: ScaleKind


discrete_format: lambda x: str
continious_format: lambda x: str


class DateTickAlgorithmKind(Enum):

    DTA_FILTER = auto()  # Compute date ticks by filtering to closest matches
    DTA_ADD_DURATION = (
        auto()
    )  # Compute date ticks by adding given duration to start time
    DTA_CUSTOM_BREAKS = auto()  # Use user-given custom breaks (as UNIX timestamps)


@dataclass
class DateScale:
    """Represents a date scale with various options for formatting and computation."""

    name: str
    ax_kind: AxisKind
    is_timestamp: bool
    breaks: List[float]
    format_string: str = None
    date_spacing: Duration = None
    date_algo: DateTickAlgorithmKind = DateTickAlgorithmKind.DTA_FILTER

    def parse_date(self, date: str):
        # todo this should return datetime
        pass


class Missing:
    pass


# Define the types
PossibleColor = Union[Missing, Color, int, str, Optional[Color]]
PossibleFloat = Union[Missing, int, float, str, Optional[float]]
PossibleBool = Union[Missing, bool]
PossibleMarker = Union[Missing, MarkerKind, Optional[MarkerKind]]
PossibleLineType = Union[Missing, LineType, Optional[LineType]]
PossibleErrorBar = Union[Missing, ErrorBarKind, Optional[ErrorBarKind]]
PossibleFont = Union[Missing, Font, Optional[Font]]
PossibleSecondaryAxis = Union[Missing, SecondaryAxis]


@dataclass
class ColorScale:
    name: str
    colors: List[int]


@dataclass
class DataKind:
    mapping: str = "mapping"
    setting: str = "setting"


@dataclass
class GGScale:
    col: FormulaNode  # The column which this scale corresponds to
    name: str  # Name of the scale
    ids: Set[int]  # Set of ids (uint16 in original, mapped to int in Python)
    value_kind: Value  # Value kind of the data of `col`
    has_discreteness: bool  # Whether discreteness is present
    num_ticks: Optional[int]  # Optional: Desired number of ticks for this scale
    breaks: Optional[List[float]]  # Optional: Position for all ticks in data units
    data_kind: DataKind  # Data kind (type of data used in this scale)
    scale_kind: ScaleKind
    discrete_kind: DiscreteKind
