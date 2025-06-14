import typing
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from python_ggplot.colormaps.color_maps import (
    INFERNO_RAW,
    MAGMARAW,
    PLASMA_RAW,
    VIRIDIS_RAW,
)
from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import (
    BLACK,
    AxisKind,
    Color,
    GGEnum,
    GGException,
    LineType,
    MarkerKind,
    Scale,
)
from python_ggplot.core.units.objects import RelativeUnit
from python_ggplot.gg.constants import DEFAULT_TO_X
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    ColumnType,
    GGValue,
    VectorCol,
    VNull,
    pandas_series_to_column,
)
from python_ggplot.gg.geom.base import Geom, GeomType, XYMinMax
from python_ggplot.gg.geom.filled_geom import (
    FilledGeom,
    FilledGeomContinuous,
    FilledGeomDiscrete,
    FilledGeomDiscreteKind,
)
from python_ggplot.gg.styles.config import (
    DEFAULT_ALPHA_RANGE_TUPLE,
    HISTO_DEFAULT_STYLE,
    LINE_DEFAULT_STYLE,
)
from python_ggplot.gg.types import (
    ContinuousFormat,
    DataType,
    DateTickAlgorithmType,
    DiscreteFormat,
    DiscreteKind,
    DiscreteType,
    SecondaryAxis,
    gg_col_anonymous,
    gg_col_const,
)
from python_ggplot.graphics.initialize import (
    InitLineInput,
    InitRectInput,
    init_line,
    init_point_from_coord,
    init_rect,
)
from python_ggplot.graphics.objects import GraphicsObject
from python_ggplot.graphics.views import ViewPort

if typing.TYPE_CHECKING:
    from python_ggplot.gg.types import GGStyle


ScaleTransformFunc = Callable[[float], float]


@dataclass
class ScaleValue(ABC):

    def __eq__(self, value: object, /) -> bool:
        # TODO Critical
        # implement or fix logic in
        #  public_interface.common.scale_x_discrete_with_labels
        # for format_discrete_label_
        return super().__eq__(value)

    @abstractmethod
    def update_style(self, style: "GGStyle"):
        pass

    @property
    @abstractmethod
    def scale_type(self) -> "ScaleType":
        pass


@dataclass
class TextScaleValue(ScaleValue):

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.TEXT


@dataclass
class SizeScaleValue(ScaleValue):
    size: Optional[float] = None

    def update_style(self, style: "GGStyle"):
        style.size = self.size

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.SIZE


@dataclass
class ShapeScaleValue(ScaleValue):
    marker: Optional[MarkerKind] = None
    line_type: Optional[LineType] = None

    def update_style(self, style: "GGStyle"):
        style.marker = self.marker
        style.line_type = self.line_type

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.SHAPE


@dataclass
class AlphaScaleValue(ScaleValue):
    alpha: Optional[float] = None

    def update_style(self, style: "GGStyle"):
        style.alpha = self.alpha

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.ALPHA


@dataclass
class FillColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.fill_color = self.color
        style.color = self.color

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.FILL_COLOR


@dataclass
class ColorScaleValue(ScaleValue):
    color: Optional[Color] = None

    def update_style(self, style: "GGStyle"):
        style.color = self.color

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.COLOR


@dataclass
class TransformedDataScaleValue(ScaleValue):
    val: Optional[Any] = None

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.TRANSFORMED_DATA


@dataclass
class LinearDataScaleValue(ScaleValue):
    val: Optional[Any] = None

    @property
    def scale_type(self) -> "ScaleType":
        return ScaleType.LINEAR_DATA


@dataclass
class ColorScale:
    name: str = ""
    colors: List[int] = field(default_factory=list)

    def __rich_repr__(self):
        """
        TODO make this generic?
        copy from GGScaleData
        """
        exclude_field = "colors"
        for field in fields(self):
            if field.name != exclude_field:
                yield field.name, getattr(self, field.name)
        # this by default would print the whole set, one item at a time
        yield f"colors -> min: {min(self.colors)} max: {max(self.colors)}"

    @classmethod
    def from_color_map(cls, name: str, color_map: List[List[float]]) -> "ColorScale":
        def _to_val(x: float):
            if x > 255 or x < 0:
                raise GGException("incorrect color")
            int_x = int(round(x * 255.0))
            return max(0, min(int_x, 255))

        colors: List[int] = []
        for r, g, b in color_map:
            new_col = (
                (255 << 24) | (_to_val(r) << 16) | (_to_val(g) << 8) | (_to_val(b))
            )
            colors.append(new_col)
        result = ColorScale(name=name, colors=colors)
        return result

    @classmethod
    def viridis(cls) -> "ColorScale":
        return cls.from_color_map("viridis", VIRIDIS_RAW)

    @classmethod
    def magmaraw(cls) -> "ColorScale":
        return cls.from_color_map("magma", MAGMARAW)

    @classmethod
    def inferno(cls) -> "ColorScale":
        return cls.from_color_map("inferno", INFERNO_RAW)

    @classmethod
    def plasma(cls) -> "ColorScale":
        return cls.from_color_map("plasma", PLASMA_RAW)


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
    def get_low_level_scale(self) -> Scale:
        pass

    @abstractmethod
    def update_filled_geom_x_attributes(
        self, fg: FilledGeom, df: pd.DataFrame, scale_col: VectorCol
    ):
        pass

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

    def update_filled_geom_x_attributes(
        self, fg: FilledGeom, df: pd.DataFrame, scale_col: VectorCol
    ):
        if not fg.gg_data.is_x_discrete():
            raise GGException("expected discrete x")

        fg.gg_data.num_x = max(fg.gg_data.num_x, scale_col.evaluate(df).nunique())
        fg.gg_data.x_scale = Scale(low=0.0, high=1.0)
        fg.gg_data.x_discrete_kind.label_seq = self.label_seq  # type: ignore

    def get_low_level_scale(self) -> Scale:
        return Scale(low=0.0, high=1.0)

    def to_filled_geom_kind(self) -> FilledGeomDiscreteKind:
        return FilledGeomDiscrete(label_seq=self.label_seq)

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


@dataclass
class GGScaleContinuous(GGScaleDiscreteKind):
    data_scale: Scale = field(default_factory=lambda: Scale(low=0.0, high=0.0))
    format_continuous_label: Optional[ContinuousFormat] = None

    def update_filled_geom_x_attributes(
        self, fg: FilledGeom, df: pd.DataFrame, scale_col: VectorCol
    ):
        if fg.geom_type != GeomType.RASTER:
            fg.gg_data.num_x = max(fg.gg_data.num_x, len(df))

    def get_low_level_scale(self) -> Scale:
        return deepcopy(self.data_scale)

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

    def get_name(self) -> Optional[str]:
        if isinstance(self.col.col_name, gg_col_const):
            return None
        if self.name:
            return self.name
        else:
            return str(self.col)

    def __rich_repr__(self):
        exclude_field = "ids"
        for field in fields(self):
            if field.name != exclude_field:
                yield field.name, getattr(self, field.name)
        # this by default would print the whole set, one item at a time
        if self.ids:
            yield f"ids -> min: {min(self.ids)} max: {max(self.ids)}"

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

    def evaluate(self, df: pd.DataFrame) -> "pd.Series[Any]":
        return self.gg_data.col.evaluate(df)

    def merge(self, other: "GGScale") -> Optional["GGScale"]:
        # TODO add validations
        # this method is experimental
        #

        if self.is_continuous() and other.is_continuous():
            self_ = isinstance(self.gg_data.col.col_name, gg_col_const)
            other_ = isinstance(other.gg_data.col.col_name, gg_col_const)
            if self_ and other_:
                result = deepcopy(self)
                result.gg_data.discrete_kind.data_scale = (
                    result.gg_data.discrete_kind.data_scale.merge(
                        other.gg_data.discrete_kind.data_scale
                    )
                )
                return result

        if self.is_discrete() and other.is_discrete():
            result = deepcopy(self)
            label_seq_self = result.gg_data.discrete_kind.label_seq
            label_seq_other = other.gg_data.discrete_kind.label_seq
            # todo make this more efficient it may be needed
            data = list(set(label_seq_self + label_seq_other))
            result.gg_data.discrete_kind.label_seq = data
            result.gg_data.col = VectorCol(col_name=gg_col_anonymous(pd.Series(data)))
            return result

        raise GGException("attempted to merge not compatible scales")

    def set_x_attributes(self, fg: FilledGeom, df: pd.DataFrame):
        self.gg_data.discrete_kind.update_filled_geom_x_attributes(
            fg, df, self.gg_data.col
        )

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

    @staticmethod
    def from_const(const_value: Any) -> "LinearDataScale":
        return LinearDataScale(
            gg_data=GGScaleData(
                col=VectorCol(col_name=gg_col_const(const_value)),
                value_kind=VNull(),
                ids=set(),
                discrete_kind=GGScaleDiscrete(label_seq=[]),
            )
        )

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


def _line_legend(
    name: str, color: Optional[Color] = None, line_width: Optional[float] = None
):
    # TODO move some logic to legends.py
    style = deepcopy(LINE_DEFAULT_STYLE)
    style.color = color
    style.line_width = line_width or 2.0
    start = Coord.relative(0.0, 0.5)
    end = Coord.relative(1.0, 0.5)
    init_line_input = InitLineInput(style=style, name=name)
    return init_line(start, end, init_line_input)


def _rect_legend(name: str, plt: ViewPort, color: Color):
    # TODO move some logic to legends.py
    style = deepcopy(HISTO_DEFAULT_STYLE)
    style.color = color
    style.fill_color = color
    origin = Coord.relative(0.05, 0.05)
    width = RelativeUnit(0.9)
    height = RelativeUnit(0.9)
    init_rect_input = InitRectInput(name=name)
    return init_rect(plt, origin, width, height, init_rect_input)


def _point_legend(
    name: str, color: Optional[Color] = None, size: Optional[float] = None
):
    # TODO move some logic to legends.py
    coord = Coord.relative(0.5, 0.5)
    return init_point_from_coord(
        coord,
        marker=MarkerKind.CIRCLE,
        color=color or deepcopy(BLACK),
        name=name,
        size=size or 3.0,
    )


def _enumerate_scale_value_map(
    scale: GGScale, access_idx: Optional[List[int]] = None
) -> Generator[tuple[GGValue, Any], Any, None]:
    (discrete_kind, idx) = discrete_legend_markers_params(scale, access_idx)
    for i in idx:
        key = discrete_kind.label_seq[i]
        val = discrete_kind.value_map[key]

        if not isinstance(val, (FillColorScaleValue, ColorScaleValue)):
            yield key, val
        else:
            if val.color is None:
                raise GGException("expected color")

            yield key, val.color


class _ColorScaleMixin(GGScale):

    def discrete_legend_markers(
        self, plt: ViewPort, geom_type: GeomType, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []
        for key, val in _enumerate_scale_value_map(self, access_idx):
            if geom_type in {GeomType.LINE, GeomType.HISTOGRAM}:
                new_go = _line_legend(str(key), val)
            elif geom_type == GeomType.TILE:
                new_go = _rect_legend(str(key), plt, val)
            else:
                new_go = _point_legend(str(key), val)

            result.append(new_go)
        return result


@dataclass
class ColorScaleKind(_ColorScaleMixin):
    color_scale: "ColorScale" = field(default_factory=ColorScale.viridis)

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
        self, plt: ViewPort, geom_type: GeomType, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []
        for key, val in _enumerate_scale_value_map(self, access_idx):
            if geom_type == GeomType.LINE:
                # improve logic from generateLegendMarkers
                # TODO high priority/easy fix this needs some overriding of the values:
                # let size = scale.getValue(scale.getLabelKey(i)).size
                # var st = LineDefaultStyle
                # st.lineWidth = size
                new_go = _line_legend(str(key), line_width=3.0)
            else:
                new_go = _point_legend(str(key))

            result.append(new_go)
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
        self, plt: ViewPort, geom_type: GeomType, access_idx: Optional[List[int]] = None
    ) -> List[GraphicsObject]:
        result: List[GraphicsObject] = []

        for key, val in _enumerate_scale_value_map(self, access_idx):
            size = val.size
            if geom_type == GeomType.LINE:
                new_go = _line_legend(str(key), line_width=size)
            else:
                new_go = _point_legend(str(key), size=size)

            result.append(new_go)
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
class MainAddScales:
    main: Optional["GGScale"] = None
    more: Optional[List["GGScale"]] = None

    def get_name(self) -> Optional[str]:
        if self.main:
            return self.main.gg_data.get_name()
        elif self.more:
            for scale in self.more:
                name = scale.gg_data.get_name()
                if name:
                    return name
        return None


@dataclass
class FilledScales:
    # TODO double check this
    # this inhertis a default value, we have to double check if its 0.0
    # but most likely it is
    x_scale: Scale = field(default_factory=lambda: Scale(low=0.0, high=0.0))
    y_scale: Scale = field(default_factory=lambda: Scale(low=0.0, high=0.0))
    reversed_x: bool = False
    reversed_y: bool = False
    discrete_x: bool = False
    discrete_y: bool = False
    geoms: List[FilledGeom] = field(default_factory=list)
    x: Optional[MainAddScales] = None
    y: Optional[MainAddScales] = None
    color: Optional[MainAddScales] = None
    fill: Optional[MainAddScales] = None
    alpha: Optional[MainAddScales] = None
    size: Optional[MainAddScales] = None
    shape: Optional[MainAddScales] = None
    x_min: Optional[MainAddScales] = None
    x_max: Optional[MainAddScales] = None
    y_min: Optional[MainAddScales] = None
    y_max: Optional[MainAddScales] = None
    xintercept: Optional[MainAddScales] = None
    yintercept: Optional[MainAddScales] = None
    width: Optional[MainAddScales] = None
    height: Optional[MainAddScales] = None
    text: Optional[MainAddScales] = None
    y_ridges: Optional[MainAddScales] = None
    weight: Optional[MainAddScales] = None
    facets: List[GGScale] = field(default_factory=list)
    metadata: Dict[Any, Any] = field(default_factory=dict)

    def get_scales_excluding_xy(self) -> List[Optional[MainAddScales]]:
        scales = [
            self.color,
            self.fill,
            self.size,
            self.shape,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.width,
            self.height,
            self.text,
            self.y_ridges,
            self.width,
            self.xintercept,
            self.yintercept,
        ]
        return scales

    def get_scale(
        self,
        attr: Optional[MainAddScales],
        geom: Optional[Geom] = None,
        optional: bool = False,
    ) -> Optional[GGScale]:
        """
        TODO low priority easy fix
        do some cleaning up here
        the optional=False/True should work
        but this forces us to return optional type so we have to do null checks
        where we know a scale is reuquired will be good to not do those
        the original logic had to macros
        genGetScale and genGetOptScale
        one returns optional and one returns non optional
        we should do the same here
        """
        if geom is None:
            geom_id = 0
        else:
            geom_id = geom.gg_data.gid

        if attr is None:
            if optional:
                return None
            raise GGException("Scale is None")

        if attr.main:
            return attr.main

        for scale in attr.more or []:
            if geom_id == 0 or geom_id in scale.gg_data.ids:
                return scale

        if optional:
            return None
        raise GGException("Scale is None")

    # TODO the following functions are repetitive
    # we can make something more re-usable
    # we keep them for now for backwards compat
    # the original ones created by macro
    def get_y_max_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.y_max, geom=geom, optional=optional)

    def get_y_min_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.y_min, geom=geom, optional=optional)

    def get_x_max_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.x_max, geom=geom, optional=optional)

    def get_x_min_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.x_min, geom=geom, optional=optional)

    def get_height_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.height, geom=geom, optional=optional)

    def get_width_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.width, geom=geom, optional=optional)

    def get_y_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.y, geom=geom, optional=optional)

    def get_x_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.x, geom=geom, optional=optional)

    def get_text_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.text, geom=geom, optional=optional)

    def get_fill_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.fill, geom=geom, optional=optional)

    def get_weight_scale(self, geom: Geom, optional: bool = False) -> Optional[GGScale]:
        return self.get_scale(attr=self.weight, geom=geom, optional=optional)

    def has_secondary(self: "FilledScales", ax_kind: AxisKind) -> bool:
        # this assumes gg_themes.has_secondary was called first to ensure it exists
        # so will raise an exception if no axis

        if AxisKind.X:
            scale = self.get_scale(self.x, optional=False)
        elif AxisKind.Y:
            scale = self.get_scale(self.y, optional=False)
        else:
            raise GGException("unexpected scale")
        if scale is None:
            print("warning scale is None")
            # TODO do we return false or raise exception?
            return False

        if not isinstance(scale, (LinearDataScale, TransformedDataScale)):
            return False

        return scale.data is not None and scale.data.secondary_axis is not None

    def get_secondary_axis(self: "FilledScales", ax_kind: AxisKind) -> "SecondaryAxis":
        """
        TODO reuse this logic with has_secondary, fine for now
        """
        if AxisKind.X:
            scale = self.get_scale(self.x, optional=False)
        elif AxisKind.Y:
            scale = self.get_scale(self.y, optional=False)
        else:
            raise GGException("unexpected scale")

        if scale is None or not isinstance(
            scale, (LinearDataScale, TransformedDataScale)
        ):
            raise GGException(
                f"Secondary axis doesnt exist for scale {scale.__class__.__name__}"
            )

        if scale.data is None:
            raise GGException(f"Scale {scale.__class__.__name__} data is none")

        if scale.data.secondary_axis is None:
            raise GGException(
                f"Scale {scale.__class__.__name__} secondary_axis is none"
            )

        return scale.data.secondary_axis

    def enumerate_scales_by_id(self: "FilledScales") -> Generator[GGScale, None, None]:
        fs_fields = [
            "x",
            "y",
            "color",
            "fill",
            "size",
            "shape",
            "yRidges",
        ]
        for fs_field in fs_fields:
            field_: Optional[MainAddScales] = getattr(self, fs_field, None)
            if field_:
                if field_.main:
                    yield field_.main
                for more_ in field_.more or []:
                    yield more_

    def enumerate_scales(
        self: "FilledScales", geom: Geom
    ) -> Generator[Any, None, None]:
        # TODO this will have to make a bunch of objects hashable
        # we may want to implement it for all
        result: Set[Any] = set()
        for scale in self.enumerate_scales_by_id():
            if geom.gg_data.gid in scale.gg_data.ids and scale not in result:
                result.add(scale)
                yield scale


def scale_from_data(
    column: "pd.Series[Any]", scale: GGScale, ignore_inf: bool = True
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


@dataclass
class CompositeScale:
    scale_min: GGScale
    scale_max: GGScale

    def get_bounds(self, df: pd.DataFrame) -> Scale:
        min = self.scale_min.evaluate(df)
        max = self.scale_min.evaluate(df)
        combined = min + max
        try:
            return Scale(low=float(combined.min()), high=float(combined.max()))
        except ValueError:
            return Scale(low=0.0, high=1.0)


T = TypeVar("T", GGScale, CompositeScale)
Y = TypeVar("Y", GGScale, CompositeScale)


@dataclass
class XYScale(ABC, Generic[T, Y]):
    primary: T
    secondary: Y
    primary_axis_kind: AxisKind

    def x_scale(self) -> GGScale | CompositeScale:
        if self.primary_axis_kind == AxisKind.X:
            return self.primary
        elif self.primary_axis_kind == AxisKind.Y:
            return self.secondary
        else:
            raise GGException("Unexpected axis kind")

    def y_scale(self) -> GGScale | CompositeScale:
        if self.primary_axis_kind == AxisKind.X:
            return self.secondary
        elif self.primary_axis_kind == AxisKind.Y:
            return self.primary
        else:
            raise GGException("Unexpected axis kind")

    @abstractmethod
    def primary_col_name(self) -> str:
        pass

    @staticmethod
    def from_geom(
        filled_cales: FilledScales, geom: Geom, xy_minmax: XYMinMax
    ) -> "XYScale[Any, Any]":
        x = filled_cales.get_x_scale(geom, optional=True)
        y = filled_cales.get_y_scale(geom, optional=True)

        x_min = filled_cales.get_x_min_scale(geom, optional=True)
        x_max = filled_cales.get_x_max_scale(geom, optional=True)
        y_min = filled_cales.get_y_min_scale(geom, optional=True)
        y_max = filled_cales.get_y_max_scale(geom, optional=True)

        # todo refactor this
        (
            x_min_from_geom,
            x_max_from_geom,
            y_min_from_geom,
            y_max_from_geom,
        ) = xy_minmax.to_scales()

        x_min = x_min_from_geom or x_min
        x_max = x_max_from_geom or x_max
        y_min = y_min_from_geom or y_min
        y_max = y_max_from_geom or y_max

        has_x_minmax = len([i for i in [x_min, x_max] if i is None]) == 0
        has_y_minmax = len([i for i in [y_min, y_max] if i is None]) == 0

        if has_y_minmax:
            y = None

        if has_x_minmax:
            x = None

        if not DEFAULT_TO_X:
            # if x and y but coord_flip=True
            # then we need primary_axis = Y
            raise GGException("support coord flip")

        if x and y:
            return XYScaleTwoColumns(
                primary=x, secondary=y, primary_axis_kind=AxisKind.X
            )

        if x is not None and has_y_minmax:
            return XYScalePrimaryComposite(
                primary=x,
                secondary=CompositeScale(scale_min=y_min, scale_max=y_max),
                primary_axis_kind=AxisKind.X,
            )
        elif y is not None and has_x_minmax:
            return XYScalePrimaryComposite(
                primary=y,
                secondary=CompositeScale(scale_min=x_min, scale_max=x_max),
                primary_axis_kind=AxisKind.Y,
            )
        elif has_x_minmax and has_y_minmax:
            return XYScaleComposite(
                primary=CompositeScale(scale_min=x_min, scale_max=x_max),
                secondary=CompositeScale(scale_min=y_min, scale_max=y_max),
                primary_axis_kind=AxisKind.X,
            )
        else:
            raise GGException(
                "Expected x and y, or x and y_min y_max or y and x_min x_max"
            )


@dataclass
class XYScaleComposite(XYScale[CompositeScale, CompositeScale]):

    def primary_col_name(self) -> str:
        raise GGException("Composite scale does not have a col name")


@dataclass
class XYScalePrimaryComposite(XYScale[GGScale, CompositeScale]):

    def primary_col_name(self) -> str:
        return str(self.primary.gg_data.col)


@dataclass
class XYScaleTwoColumns(XYScale[GGScale, GGScale]):

    def primary_col_name(self) -> str:
        return str(self.primary.gg_data.col)
