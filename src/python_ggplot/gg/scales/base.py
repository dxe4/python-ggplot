import typing
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from python_ggplot.colormaps.color_maps import VIRIDIS_RAW_COLOR_SCALE
from python_ggplot.core.objects import AxisKind, GGEnum, GGException, MarkerKind, Point, Scale
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    ColumnType,
    GGValue,
    VectorCol,
    pandas_series_to_column,
)
from python_ggplot.gg.geom import (
    FilledGeom,
    FilledGeomContinuous,
    FilledGeomDiscrete,
    FilledGeomDiscreteKind,
    Geom,
)
from python_ggplot.gg.scales.values import (
    ColorScaleValue,
    FillColorScaleValue,
    ShapeScaleValue,
    SizeScaleValue,
)
from python_ggplot.gg.styles import DEFAULT_ALPHA_RANGE_TUPLE
from python_ggplot.gg.types import (
    ContinuousFormat,
    DataType,
    DateTickAlgorithmType,
    DiscreteFormat,
    DiscreteKind,
    DiscreteType,
    MainAddScales,
    SecondaryAxis,
)
from python_ggplot.graphics.initialize import init_point_from_point
from python_ggplot.graphics.objects import GraphicsObject
from python_ggplot.graphics.views import ViewPort

if typing.TYPE_CHECKING:
    from python_ggplot.gg.scales.values import ScaleValue
    from python_ggplot.gg.types import GGStyle


# TODO port those 2 macros, wont port until they are needed
# macro genGetOptScale
# macro genGetScale

ScaleTransformFunc = Callable[[float], float]


@dataclass
class ColorScale:
    name: str = ""
    colors: List[int] = field(default_factory=list)


class ScaleType(GGEnum):
    LINEAR_DATA = auto()
    TRANSFORMED_DATA = auto()
    COLOR = auto()
    FILL_COLOR = auto()
    ALPHA = auto()
    SHAPE = auto()
    SIZE = auto()
    TEXT = auto()


@dataclass
class LinearAndTransformScaleData:
    axis_kind: AxisKind = AxisKind.X
    reversed: bool = False
    # TODO high priority
    # change the lambda to print a warning or raise an exception
    transform: ScaleTransformFunc = field(default=lambda x: x)
    secondary_axis: Optional["SecondaryAxis"] = None
    date_scale: Optional["DateScale"] = None


class GGScaleDiscreteKind(DiscreteKind, ABC):

    @abstractmethod
    def to_filled_geom_kind(self) -> FilledGeomDiscreteKind:
        pass

    @property
    @abstractmethod
    def discrete_type(self) -> DiscreteType:
        pass


@dataclass
class GGScaleDiscrete(GGScaleDiscreteKind):
    value_map: OrderedDict[GGValue, "ScaleValue"] = field(default_factory=OrderedDict)
    label_seq: List[GGValue] = field(default_factory=list)
    format_discrete_label: Optional[DiscreteFormat] = None

    def to_filled_geom_kind(self) -> FilledGeomDiscreteKind:
        return FilledGeomDiscrete(label_seq=self.label_seq)

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


@dataclass
class GGScaleContinuous(GGScaleDiscreteKind):
    data_scale: Scale = field(default_factory=lambda: Scale(low=0.0, high=0.0))
    format_continuous_label: Optional[ContinuousFormat] = None

    def to_filled_geom_kind(self) -> FilledGeomDiscreteKind:
        return FilledGeomContinuous()

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.CONTINUOUS

    def map_data(self) -> List["ScaleValue"]:
        # TODO does this need to be a param or static func is fune?
        raise GGException("todo")


@dataclass
class GGScaleData:
    col: VectorCol
    value_kind: GGValue
    ids: Set[int] = field(default_factory=set)
    has_discreteness: bool = False
    # I dont like this default, but copying the origin for now
    data_type: DataType = DataType.MAPPING
    discrete_kind: "GGScaleDiscreteKind" = field(default_factory=GGScaleDiscrete)
    num_ticks: Optional[int] = None
    breaks: Optional[List[float]] = None
    name: str = ""

    @staticmethod
    def create_empty_scale(col: str = "") -> "GGScaleData":
        # TODO i really dont like this but its how is done
        # sticking to the convention for now
        # but moving in a centralised place
        return GGScaleData(col=VectorCol(col), value_kind=VTODO())


@dataclass
class GGScale(ABC):
    """
    TODO make some of the nested data available here
    eg scale.gg_data.discrete_kind.discrete_type -> scale.discrete_type
    + rename gg_data before alpha
    """

    gg_data: GGScaleData

    def assign_breaks(self, breaks: Union[int, List[float]]) -> None:
        """
        TODO we need to make sure the types work for numpy...
        """
        if isinstance(breaks, int):
            self.gg_data.num_ticks = breaks
        elif all(isinstance(x, float) for x in breaks):
            self.gg_data.breaks = breaks
        else:
            self.gg_data.breaks = [float(x) for x in breaks]

    def get_col_name(self: "GGScale") -> str:
        if self.scale_type == ScaleType.TRANSFORMED_DATA:
            # scale.col.evaluate()
            # TODO: This falls into datamancer / pandas compatibility
            # it will eventually fall into place, but for now we have to keep as is until the rest is working
            scale_name = str(self)
            return f"log10({scale_name})"
        else:
            # scale.col.evaluate()  TODO: same here
            return str(self.gg_data.col)

    @property
    @abstractmethod
    def scale_type(self) -> ScaleType:
        pass

    def is_discrete(self) -> bool:
        return self.gg_data.discrete_kind.discrete_type == DiscreteType.DISCRETE

    def is_continuous(self) -> bool:
        return self.gg_data.discrete_kind.discrete_type == DiscreteType.CONTINUOUS

    def is_reversed(self) -> bool:
        if isinstance(self, (LinearDataScale, TransformedDataScale)):
            if self.data is None:
                return False
            return self.data.reversed
        return False

    # def __eq__(self, other) -> bool:  # type: ignore
    #     return (
    #         self.discrete_kind == other.discrete_kind
    #         and self.col.name == other.col.name
    #     )


@dataclass
class DateScale(GGScale):
    name: str
    axis_kind: AxisKind
    is_timestamp: bool
    breaks: List[float]
    format_string: str
    date_spacing: timedelta
    date_algo: DateTickAlgorithmType = DateTickAlgorithmType.FILTER

    def parse_date(self, date: str) -> datetime:
        # TODO high priority this should return datetime
        # TODO sanity check down the line, do we allow this being dynamic?
        raise GGException("implement me")


@dataclass
class LinearDataScale(GGScale):
    transform: Optional[VectorCol] = None  # for SecondaryAxis
    data: Optional[LinearAndTransformScaleData] = None

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.LINEAR_DATA


def _default_trans(x: float) -> float:
    """
    TODO shall we just raise an exception here?
    if not then change to logger.warn?
    """
    print("warning you are using default transform which does nothing")
    return x


def _default_inverse_trans(x: float) -> float:
    """
    TODO shall we just raise an exception here?
    if not then change to logger.warn?
    """
    print("warning you are using default transform which does nothing")
    return x


@dataclass
class TransformedDataScale(GGScale):
    data: Optional[LinearAndTransformScaleData] = None
    transform: ScaleTransformFunc = _default_trans
    inverse_transform: ScaleTransformFunc = _default_inverse_trans

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TRANSFORMED_DATA

    @staticmethod
    def defualt_trans() -> ScaleTransformFunc:
        return _default_trans

    @staticmethod
    def defualt_inverse_trans() -> ScaleTransformFunc:
        return _default_trans


def discrete_legend_markers_params(
    scale: GGScale, access_idx: Optional[List[int]] = None
) -> Tuple[GGScaleDiscrete, List[int]]:
    discrete_kind = scale.gg_data.discrete_kind

    if not isinstance(discrete_kind, GGScaleDiscrete):
        raise GGException("expected discrete scale")

    if access_idx is None:
        idx = list(range(len(discrete_kind.value_map)))
    else:
        idx = access_idx

    if len(idx) != len(discrete_kind.value_map):
        raise GGException(
            f"Custom ordering of legend keys must assign each key only once! "
            f"Assigned keys: {access_idx} for num keys: {len(discrete_kind.value_map)}"
        )
    return discrete_kind, idx


class _ColorScaleMixin(GGScale):

    def discrete_legend_markers(
        self, plt: ViewPort, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []
        (discrete_kind, idx) = discrete_legend_markers_params(self, access_idx)
        for i in idx:
            key = discrete_kind.label_seq[i]
            val = discrete_kind.value_map[key]

            if not isinstance(val, (FillColorScaleValue, ColorScaleValue)):
                raise GGException("expected value of color")
            if val.color is None:
                raise GGException("expected color")

            new_point = init_point_from_point(
                plt,
                Point(0.0, 0.0),
                marker=MarkerKind.CIRCLE,
                color=val.color,
                name=str(key),
            )
            result.append(new_point)
        return result


@dataclass
class ColorScaleKind(_ColorScaleMixin):
    color_scale: "ColorScale" = VIRIDIS_RAW_COLOR_SCALE

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.COLOR


@dataclass
class FillColorScale(_ColorScaleMixin):
    color_scale: "ColorScale"

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.FILL_COLOR


@dataclass
class AlphaScale(GGScale):
    alpha: float = field(default=0.0)
    # TODO: cirtical
    # this got lost in translation, we need to revive it
    # check all usage of alpha scale and pass alpha scale instead of alpha
    alpha_range: Tuple[float, float] = DEFAULT_ALPHA_RANGE_TUPLE

    def update_style(self, style: "GGStyle"):
        style.alpha = self.alpha

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.ALPHA


@dataclass
class ShapeScale(GGScale):

    def discrete_legend_markers(
        self, plt: ViewPort, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []
        (discrete_kind, idx) = discrete_legend_markers_params(self, access_idx)
        for i in idx:
            key = discrete_kind.label_seq[i]
            val = discrete_kind.value_map[key]

            if not isinstance(val, ShapeScaleValue):
                raise GGException("expected value of shape")
            if val.marker is None:
                raise GGException("expected color")

            new_point = init_point_from_point(
                plt, Point(0.0, 0.0), marker=val.marker, name=str(key)
            )
            result.append(new_point)
        return result

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SHAPE


@dataclass
class SizeScale(GGScale):
    # low and high
    size: SizeScaleValue = field(default_factory=SizeScaleValue)
    # TODO
    # this is low Low,High its not very intuitive
    # if you dont know what it already does
    # refactor later
    size_range: Tuple[float, float] = field(default=(0.0, 0.0))

    def discrete_legend_markers(
        self, plt: ViewPort, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []
        (discrete_kind, idx) = discrete_legend_markers_params(self, access_idx)
        for i in idx:
            key = discrete_kind.label_seq[i]
            val = discrete_kind.value_map[key]

            if not isinstance(val, SizeScaleValue):
                raise GGException("expected value of size")
            if val.size is None:
                raise GGException("expected color")

            new_point = init_point_from_point(
                plt,
                Point(0.0, 0.0),
                marker=MarkerKind.CIRCLE,
                size=val.size,
                name=str(key),
            )
            result.append(new_point)
        return result

    def update_style(self, style: "GGStyle"):
        # TODO bad naming
        style.size = self.size.size

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.SIZE


@dataclass
class TextScale(GGScale):

    @property
    def scale_type(self) -> ScaleType:
        return ScaleType.TEXT


class AbstractGGScale(GGScale):
    pass


class ScaleFreeKind(GGEnum):
    FIXED = auto()
    FREE_X = auto()
    FREE_Y = auto()
    FREE = auto()


@dataclass
class FilledScales:
    x_scale: GGScale
    y_scale: GGScale
    reversed_x: bool
    reversed_y: bool
    discrete_x: bool
    discrete_y: bool
    geoms: List[FilledGeom]
    x: MainAddScales
    y: MainAddScales
    color: MainAddScales
    fill: MainAddScales
    alpha: MainAddScales
    size: MainAddScales
    shape: MainAddScales
    x_min: MainAddScales
    x_max: MainAddScales
    y_min: MainAddScales
    y_max: MainAddScales
    width: MainAddScales
    height: MainAddScales
    text: MainAddScales
    y_ridges: MainAddScales
    weight: MainAddScales
    facets: List[GGScale]
    metadata: Dict[Any, Any] = field(default_factory=dict)

    # TODO high priorotiy those are generated by macro
    # we need to port this logic of the macro too
    # for now this will do, but is urgent to port the rest
    # for s in filledScales.`field`.more:
    #   if geom.gid == 0 or geom.gid in s.ids:
    #     return s
    def get_x_scale(self: "FilledScales") -> GGScale:
        return self.x_scale

    def get_y_scale(self: "FilledScales") -> GGScale:
        return self.y_scale

    def has_secondary(self: "FilledScales", ax_kind: AxisKind) -> "SecondaryAxis":
        # this assumes gg_themes.has_secondary was called first to ensure it exists
        # so will raise an exception if no axis

        scale_getters = {
            AxisKind.X: self.x_scale,
            AxisKind.Y: self.y_scale,
        }
        gg_scale: GGScale = scale_getters[ax_kind]
        # TODO medium priority, same as `get_secondary_axis`
        if gg_scale.scale_kind.data.secondary_axis is None:  # type: ignore
            raise GGException("secondary_axis does not exist")
        return gg_scale.scale_kind.data.secondary_axis  # type: ignore

    def get_secondary_axis(self: "FilledScales", ax_kind: AxisKind) -> "SecondaryAxis":
        # this assumes gg_themes.has_secondary was called first to ensure it exists
        # so will raise an exception if no axis
        scale_getters = {
            AxisKind.X: self.x_scale,
            AxisKind.Y: self.y_scale,
        }

        gg_scale: GGScale = scale_getters[ax_kind]
        # TODO medium priority if we had FilledScales[LinearData]
        # there would be no type error
        # for now, assume correct type is passed in and fix later
        if gg_scale.scale_kind.data.secondary_axis is None:  # type: ignore
            raise GGException("secondary_axis does not exist")
        return gg_scale.scale_kind.data.secondary_axis  # type: ignore

    def enumerate_scales_by_id(self: "FilledScales") -> Generator[Any]:
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
            field_ = getattr(self, field, None)  # type: ignore
            if field_:
                yield field_
                # TODO sanity check that i understand the logic of .more in nim
                # cant seem to find docs for it, it maybe a template in the codebase
                for _, sub_field in asdict(field_).items():
                    yield sub_field

    def enumerate_scales(self: "FilledScales", geom: Geom) -> Generator[Any]:
        # TODO this will have to make a bunch of objects hashable
        # we may want to implement it for all
        result: Set[Any] = set()
        for scale in self.enumerate_scales_by_id():
            if geom.gg_data.gid in scale.ids and scale not in result:
                result.add(scale)
                yield scale


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
                f"The input column `{scale.gg_data.col}` is constant Cannot compute a numeric scale from it."
            )

    elif column_type in [ColumnType.BOOL, ColumnType.STRING, ColumnType.NONE]:
        raise ValueError(
            f"The input column `{scale.gg_data.col}` is of kind {column_type} and thus discrete. "
            "`scale_from_data` should never be called."
        )

    elif column_type == ColumnType.GENERIC:
        raise ValueError(
            f"The input column `{scale.gg_data.col}` is of kind {column.kind}. "
            "Generic columns are not supported yet."
        )

    return Scale(low=0.0, high=0.0)


def enable_scales_by_id_vega():
    # TODO vega is not supported at stage 1
    raise GGException("Vega not supported yet")


def scale_type_to_cls(scale_type: ScaleType) -> Type[GGScale]:
    data: Dict[ScaleType, Type[GGScale]] = {
        ScaleType.LINEAR_DATA: LinearDataScale,
        ScaleType.TRANSFORMED_DATA: TransformedDataScale,
        ScaleType.COLOR: ColorScaleKind,
        ScaleType.FILL_COLOR: FillColorScale,
        ScaleType.ALPHA: AlphaScale,
        ScaleType.SHAPE: ShapeScale,
        ScaleType.SIZE: SizeScale,
        ScaleType.TEXT: TextScale,
    }
    return data[scale_type]
