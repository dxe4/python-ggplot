import typing
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any, Generator, List, Optional, OrderedDict, Set, Tuple

import numpy as np
import pandas as pd

from python_ggplot.core.objects import (
    AxisKind,
    Color,
    Duration,
    GGException,
    LineType,
    MarkerKind,
    Scale,
)
from python_ggplot.datamancer_pandas_compat import (
    ColumnType,
    FormulaNode,
    GGValue,
    VNull,
    pandas_series_to_column,
)
from python_ggplot.gg_drawing import gg_draw
from python_ggplot.gg_geom import FilledGeom, Geom
from python_ggplot.gg_types import (
    ContinuousFormat,
    DataKind,
    DateTickAlgorithmKind,
    DiscreteFormat,
    DiscreteKind,
    DiscreteType,
    GgPlot,
    SecondaryAxis,
)

if typing.TYPE_CHECKING:
    from python_ggplot.gg_geom import FilledScales
    from python_ggplot.gg_types import GGStyle


# TODO port those 2 macros, wont port until they are needed
# macro genGetOptScale
# macro genGetScale


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
    def scale_type(self) -> ScaleType:
        pass


@dataclass
class ColorScaleData:
    color_scale: "ColorScale"


@dataclass
class LinearAndTransformScaleData:
    axis_kind: AxisKind
    reversed: bool
    transform: ScaleTransform
    secondary_axis: Optional["SecondaryAxis"] = None
    date_scale: Optional["DateScale"] = None


@dataclass
class LinearDataScale(ScaleKind):
    transform: Optional[FormulaNode] = None  # for SecondaryAxis
    data: Optional[LinearAndTransformScaleData] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.LINEAR_DATA


@dataclass
class TransformedDataScale(ScaleKind):
    data: Optional[LinearAndTransformScaleData] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TRANSFORMED_DATA

    def transform(self):
        pass

    def inverse_transform(self):
        pass


class ColorScaleKind(ScaleKind):
    data: Optional[ColorScaleData] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.COLOR


class FillColorScale(ScaleKind):
    data: Optional[ColorScaleData] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.FILL_COLOR


class AlphaScale(ScaleKind):
    alpha: float

    def update_style(self, style: "GGStyle"):
        style.alpha = self.alpha

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.ALPHA


class ShapeScale(ScaleKind):

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SHAPE


class SizeScale(ScaleKind):
    # low and high
    size_range = Tuple[float, float]
    size: float

    def update_style(self, style: "GGStyle"):
        style.size = self.size

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SIZE


class TextScale(ScaleKind):

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TEXT


@dataclass
class ScaleValue(ABC):

    @abstractmethod
    def update_style(self, style: "GGStyle"):
        pass

    @property
    @abstractmethod
    def scale_type(self) -> ScaleType:
        pass


class TextScaleValue(ScaleValue):

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TEXT


class SizeScaleValue(ScaleValue):
    size: Optional[float] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SIZE


class ShapeScaleValue(ScaleValue):
    marker: Optional[MarkerKind] = None
    line_type: Optional[LineType] = None

    def update_style(self, style: "GGStyle"):
        style.marker = self.marker
        style.line_type = self.line_type

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SHAPE


class AlphaScaleValue(ScaleValue):
    alpha: Optional[float] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.ALPHA


class FillColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.fill_color = self.color
        style.color = self.color

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.FILL_COLOR


class ColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.color = self.color

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.COLOR


class TransformedDataScaleValue(ScaleValue):
    val: Optional[GGValue] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TRANSFORMED_DATA


class LinearDataScaleValue(ScaleValue):
    val: Optional[GGValue] = None

    @property
    def scale_type(self) -> ScaleType:
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

    def map_data(self) -> List["ScaleValue"]:
        # TODO does this need to be a param or static func is fune?
        raise GGException("todo")


@dataclass
class GGScale:
    col: FormulaNode
    ids: Set[int]
    value_kind: GGValue
    has_discreteness: bool
    data_kind: DataKind
    scale_kind: ScaleKind
    discrete_kind: GGScaleDiscreteKind
    num_ticks: Optional[int] = None
    breaks: Optional[List[float]] = None
    name: str = ""

    def __eq__(self, other: "GGScale") -> bool:
        return (
            self.discrete_kind == other.discrete_kind
            and self.col.name == other.col.name
        )


class ScaleFreeKind(Enum):
    FIXED = auto()
    FREE_X = auto()
    FREE_Y = auto()
    FREE = auto()


def scale_from_data(
    column: pd.Series[Any], scale: GGScale, ignore_inf: bool = True
) -> Scale:
    if column.len == 0:
        return Scale(low=0.0, high=0.0)

    column_type = pandas_series_to_column(column)

    if column_type in [ColumnType.FLOAT, ColumnType.INT, ColumnType.OBJECT]:
        t = column.dropna()
        if len(t) == 0:
            return Scale(low=0.0, high=0.0)

        if ignore_inf:
            t = t[~np.isinf(t)]
            if len(t) == 0:
                return Scale(low=0.0, high=0.0)

        return Scale(low=float(t.min()), high=float(t.max()))  # type: ignore

    elif len(column.unique()) == 1:  # type: ignore
        # TODO i think this case is a bit different in pandas.
        # but keep it simple for now
        if not column.empty and column_type in [ColumnType.INT, ColumnType.FLOAT]:
            val = float(column.iloc[0])  # type: ignore
            return Scale(low=val, high=val)
        else:
            raise ValueError(
                f"The input column `{scale.col}` is constant Cannot compute a numeric scale from it."
            )

    elif column_type in [ColumnType.BOOL, ColumnType.STRING, ColumnType.NONE]:
        raise ValueError(
            f"The input column `{scale.col}` is of kind {column_type} and thus discrete. "
            "`scale_from_data` should never be called."
        )

    elif column_type == ColumnType.GENERIC:
        raise ValueError(
            f"The input column `{scale.col}` is of kind {column.kind}. "
            "Generic columns are not supported yet."
        )

    return Scale(low=0.0, high=0.0)


def get_col_name(scale: GGScale) -> str:
    if scale.scale_kind.scale_type == ScaleType.TRANSFORMED_DATA:
        # scale.col.evaluate()
        # TODO: This falls into datamancer / pandas compatibility
        # it will eventually fall into place, but for now we have to keep as is until the rest is working
        scale_name = str(scale)
        return f"log10({scale_name})"
    else:
        # scale.col.evaluate()  TODO: same here
        return str(scale.col)


def enumerate_scales_by_id(filled_scales: "FilledScales") -> Generator[Any]:
    fields = [
        "x",
        "y",
        "color",
        "fill",
        "size",
        "shape",
        "yRidges",
    ]
    for field in fields:
        field_ = getattr(filled_scales, field, None)  # type: ignore
        if field_:
            yield field_
            # TODO sanity check that i understand the logic of .more in nim
            # cant seem to find docs for it, it maybe a template in the codebase
            for _, sub_field in asdict(field_).items():
                yield sub_field


def enable_scales_by_id_vega():
    # TODO vega is not supported at stage 1
    raise GGException("Vega not supported yet")


def enumerate_scales(filled_scales: FilledScales, geom: Geom) -> Generator[Any]:
    # TODO this will have to make a bunch of objects hashable
    # we may want to implement it for all
    result: Set[Any] = set()
    for scale in enumerate_scales_by_id(filled_scales):
        if geom.gid in scale.ids and scale not in result:
            result.add(scale)
            yield scale


def update_aes_ridges(plot: GgPlot) -> "GgPlot":
    if plot.ridges is None:
        raise GGException("expected ridges")

    ridge = plot.ridges
    data = LinearAndTransformScaleData(
        # TODO, reversed and transform are required,
        # but update ridges doesn't explicitly set them
        axis_kind=AxisKind.Y,
        reversed=False,
        transform=ScaleTransform(),
    )
    scale = GGScale(
        scale_kind=LinearDataScale(data=data),
        col=ridge.col,
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(),
        ids=set(range(65536)),
        data_kind=DataKind(),
        value_kind=VNull(),
    )

    plot.aes.y_ridges = scale
    return plot


def get_secondary_axis(
    filled_scales: FilledScales, ax_kind: AxisKind
) -> "SecondaryAxis":
    # this assumes gg_themes.has_secondary was called first to ensure it exists
    # so will raise an exception if no axis
    scale_getters = {
        AxisKind.X: filled_scales.x_scale,
        AxisKind.Y: filled_scales.y_scale,
    }

    gg_scale: GGScale = scale_getters[ax_kind]
    # TODO medium priority if we had FilledScales[LinearData]
    # there would be no type error
    # for now, assume correct type is passed in and fix later
    if gg_scale.scale_kind.data.secondary_axis is None:  # type: ignore
        raise GGException("secondary_axis does not exist")
    return gg_scale.scale_kind.data.secondary_axis  # type: ignore


def has_secondary(filled_scales: FilledScales, ax_kind: AxisKind) -> "SecondaryAxis":
    # this assumes gg_themes.has_secondary was called first to ensure it exists
    # so will raise an exception if no axis

    scale_getters = {
        AxisKind.X: filled_scales.x_scale,
        AxisKind.Y: filled_scales.y_scale,
    }
    gg_scale: GGScale = scale_getters[ax_kind]
    # TODO medium priority, same as `get_secondary_axis`
    if gg_scale.scale_kind.data.secondary_axis is None:  # type: ignore
        raise GGException("secondary_axis does not exist")
    return gg_scale.scale_kind.data.secondary_axis  # type: ignore


# TODO high priorotiy those are generated by macro
# we need to port this logic of the macro too
# for now this will do, but is urgent to port the rest
# for s in filledScales.`field`.more:
#   if geom.gid == 0 or geom.gid in s.ids:
#     return s
def get_x_scale(filled_scales: FilledScales) -> GGScale:
    return filled_scales.x_scale


def get_y_scale(filled_scales: FilledScales) -> GGScale:
    return filled_scales.x_scale
