from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import auto
from math import isclose
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import pandas as pd

from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import GGEnum, GGException, Scale, Style
from python_ggplot.core.units.objects import DataUnit
from python_ggplot.gg.datamancer_pandas_compat import VNull
from python_ggplot.gg.styles.config import (
    AREA_DEFAULT_STYLE,
    BAR_DEFAULT_STYLE,
    HISTO_DEFAULT_STYLE,
    POINT_DEFAULT_STYLE,
    RECT_DEFAULT_STYLE,
    TEXT_DEFAULT_STYLE,
    TILE_DEFAULT_STYLE,
    default_line_style,
)
from python_ggplot.gg.types import (
    Aesthetics,
    BinByType,
    BinPositionType,
    GGStyle,
    PositionType,
    StatBin,
    StatKind,
    StatType,
    gg_col_const,
)
from python_ggplot.graphics.initialize import (
    InitLineInput,
    InitRectInput,
    InitTextInput,
    init_line,
    init_point,
    init_rect,
    init_text,
)
from python_ggplot.graphics.objects import GraphicsObject
from python_ggplot.graphics.views import ViewPort
from tests.test_view import AxisKind, RelativeCoordType

if TYPE_CHECKING:
    from python_ggplot.gg.geom.filled_geom import FilledGeom
    from python_ggplot.gg.scales.base import CompositeScale, GGScale


class HistogramDrawingStyle(GGEnum):
    BARS = auto()
    OUTLINE = auto()


class GeomType(GGEnum):
    POINT = auto()
    BAR = auto()
    HISTOGRAM = auto()
    FREQ_POLY = auto()
    TILE = auto()
    LINE = auto()
    ERROR_BAR = auto()
    TEXT = auto()
    RASTER = auto()
    GEOM_AREA = auto()
    GEOM_VLINE = auto()
    GEOM_HLINE = auto()
    GEOM_ABLINE = auto()
    GEOM_RECT = auto()


@dataclass
class GeomData:
    gid: int
    stat_kind: StatKind
    data: Optional[pd.DataFrame] = None
    user_style: Optional[GGStyle] = None
    position: Optional[PositionType] = None
    aes: Aesthetics = field(default_factory=Aesthetics)
    bin_position: Optional[BinPositionType] = None
    # used for geom_type histogram
    histogram_drawing_style: Optional[HistogramDrawingStyle] = None


@dataclass
class Geom(ABC):
    gg_data: GeomData

    @staticmethod
    def assign_bin_fields(
        geom: "Geom",
        st_kind: StatType,
        bins: int,
        bin_width: float,
        breaks: List[float],
        bin_by_type: BinByType,
        density: bool,
    ):
        stat_kind = geom.gg_data.stat_kind
        if isinstance(stat_kind, StatBin):
            if len(breaks) > 0:
                stat_kind.bin_edges = breaks
            if bin_width > 0.0:
                stat_kind.bin_width = bin_width
            if bins > 0:
                stat_kind.num_bins = bins
            stat_kind.bin_by = bin_by_type
            stat_kind.density = density

    @abstractmethod
    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        pass

    @property
    @abstractmethod
    def geom_type(self) -> GeomType:
        pass

    @property
    def stat_type(self) -> StatType:
        return self.gg_data.stat_kind.stat_type

    @property
    @abstractmethod
    def allowed_stat_types(self) -> List["StatType"]:
        pass

    @abstractmethod
    def default_style(self) -> Style:
        pass

    def has_bars(self) -> bool:
        # can be true for for geom_bar and geom_histogram
        return False

    def inherit_aes(self) -> bool:
        return True


class GeomPoint(Geom):

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]

    def default_style(self) -> Style:
        return deepcopy(POINT_DEFAULT_STYLE)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.POINT

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        new_point = init_point(pos, style)
        view.add_obj(new_point)


class GeomRectDrawMixin:
    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        from python_ggplot.gg.drawing import read_or_calc_bin_width

        if fg.gg_data.x_col is None:
            raise GGException("x_col does not exist")
        if fg.gg_data.x_discrete_kind is None:
            raise GGException("x_discrete_kind does not exist")

        bin_width = read_or_calc_bin_width(
            df, idx, fg.gg_data.x_col, dc_kind=fg.gg_data.x_discrete_kind.discrete_type
        )

        if bin_width != bin_widths[0]:
            raise GGException("Invalid bin width generated")

        if y is None or pd.isna(y):
            y = 0.0

        new_rect = init_rect(
            view,
            pos,
            DataUnit(bin_width),
            DataUnit(-y),
            InitRectInput(style=style, name="geom_bar_rect"),
        )

        if isclose(y, 0.0):
            # with fill/count we generate empty 0 sized elements for every combo
            # we shouldn't really deal with this here, but outside
            # for now its fine
            print(
                "WARNING: trying to render a rect of height 0.0. this can happen with fill/stack"
            )
        else:
            view.add_obj(new_rect)


class GeomHistogramMixin(GeomRectDrawMixin):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.HISTOGRAM


class GeomBarMixin(GeomRectDrawMixin):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.BAR


class GeomBar(GeomRectDrawMixin, Geom):

    def has_bars(self) -> bool:
        return True

    def default_style(self) -> Style:
        return deepcopy(BAR_DEFAULT_STYLE)

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [StatType.IDENTITY, StatType.COUNT]

    @property
    def geom_type(self) -> GeomType:
        return GeomType.BAR


@dataclass
class GeomHistogram(GeomHistogramMixin, Geom):
    histogram_drawing_style: HistogramDrawingStyle

    def has_bars(self) -> bool:
        return self.histogram_drawing_style == HistogramDrawingStyle.BARS

    def default_style(self) -> Style:
        return deepcopy(HISTO_DEFAULT_STYLE)

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [StatType.IDENTITY, StatType.BIN, StatType.DENSITY]

    @property
    def geom_type(self) -> GeomType:
        return GeomType.HISTOGRAM


class GeomFreqPoly(Geom):

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [StatType.IDENTITY, StatType.BIN, StatType.DENSITY]

    @property
    def geom_type(self) -> GeomType:
        return GeomType.FREQ_POLY

    def default_style(self):
        return default_line_style(self.stat_type)

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


class GeomErrorBarMixin:
    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        from python_ggplot.gg.drawing import draw_error_bar
        from python_ggplot.gg.geom.filled_geom import FilledGeomErrorBar

        temp_fg = cast(FilledGeomErrorBar, fg)
        new_error_bar = draw_error_bar(view, temp_fg, pos, df, idx, style)
        view.add_obj(new_error_bar)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.ERROR_BAR


@dataclass
class GeomErrorBar(GeomErrorBarMixin, Geom):
    xy_minmax: "XYMinMax"

    def default_style(self):
        return default_line_style(self.stat_type)

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]


class GeomTextMixin:
    @property
    def geom_type(self) -> GeomType:
        return GeomType.TEXT

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        from python_ggplot.gg.drawing import read_text

        if style.font is None:
            raise GGException("expected style.font")

        new_text = init_text(
            view,
            pos,
            InitTextInput(
                text=read_text(df, idx, fg),
                align_kind=style.font.align_kind,
                font=style.font,
                # TODO high priority seems ggplot is using a var we arent aware
                # for now we just proceed without it
                # text_kind="text"
            ),
        )
        view.add_obj(new_text)


class GeomText(GeomTextMixin, Geom):

    def default_style(self) -> Style:
        return deepcopy(TEXT_DEFAULT_STYLE)

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]


class GeomRasterMixin:
    @property
    def geom_type(self) -> GeomType:
        return GeomType.RASTER

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


class GeomRaster(GeomRasterMixin, Geom):

    def default_style(self) -> Style:
        raise GGException("Rraster does not have default style")

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]


class GeomTileMixin:
    @property
    def geom_type(self) -> GeomType:
        return GeomType.TILE

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        new_rect = init_rect(
            view,
            pos,
            DataUnit(bin_widths[0]),
            DataUnit(-bin_widths[1]),
            InitRectInput(style=style),
        )
        view.add_obj(new_rect)


class GeomTile(GeomTileMixin, Geom):

    def default_style(self) -> Style:
        return deepcopy(TILE_DEFAULT_STYLE)

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]


class StaticLine(ABC):

    @abstractmethod
    def axis_kind(self) -> AxisKind:
        pass

    @abstractmethod
    def intercept_field(
        self,
    ) -> Optional[Union[Union[float, int], Iterable[Union[float, int]]]]:
        pass

    @abstractmethod
    def get_scale(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        series: Optional["pd.Series[Any]"] = None,
    ) -> Optional[Scale]:
        pass

    def values_to_use(
        self,
        series: Optional["pd.Series[Any]"] = None,
    ) -> List[float]:
        intercept_field = self.intercept_field()
        if intercept_field is not None:
            if isinstance(intercept_field, Iterable):
                # str is iterable and if that happens it will cause issue
                return list(intercept_field)
            else:
                return [intercept_field]
        elif series is not None:
            return list(series)
        else:
            # this shouldn't happen but making sure we have an error in case is needed
            raise GGException("expected either an intercept value or a column")

    def _draw_static_line(
        self,
        val: Any,
        scale: Scale,
        style: Style,
        axis_kind: AxisKind,
        series: Optional["pd.Series[Any]"] = None,
    ) -> GraphicsObject:
        if scale.low == 0.0 and scale.high == 1.0 and series is not None:
            indices = series.loc[series == val].index.to_list()
            if len(indices) > 1:
                # known issue fine for now
                raise GGException("TODO: support multiple indices for vline")
            rel_pos = (indices[0]) / len(indices)
        else:
            rel_pos = (float(val) - scale.low) / (scale.high - scale.low)
        if axis_kind == AxisKind.X:
            x = RelativeCoordType(rel_pos)
            y1 = RelativeCoordType(0.0)
            y2 = RelativeCoordType(1.0)
            start = Coord(x=x, y=y1)
            end = Coord(x=x, y=y2)
        elif axis_kind == AxisKind.Y:
            y = RelativeCoordType(1 - rel_pos)
            x1 = RelativeCoordType(0.0)
            x2 = RelativeCoordType(1.0)
            start = Coord(x=x1, y=y)
            end = Coord(x=x2, y=y)
        else:
            raise GGException("Unepected axis")

        line = init_line(
            start,
            end,
            InitLineInput(style=style),
        )
        return line

    def draw_detached_geom(
        self,
        view: ViewPort,
        filled_geom: "FilledGeom",
        style: Style,
        series: Optional["pd.Series[Any]"] = None,
    ):
        scale = self.get_scale(view, filled_geom, series)
        if scale is None:
            raise GGException("expected a scale to draw static line")

        for position in self.values_to_use(series):
            line = self._draw_static_line(
                position, scale, style, self.axis_kind(), series
            )
            view.children[0].objects.append(line)

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


@dataclass
class GeomHLine(StaticLine, Geom):
    yintercept: Optional[Union[Union[float, int], Iterable[Union[float, int]]]]
    inhert_aes: bool = False

    def axis_kind(self) -> AxisKind:
        return AxisKind.Y

    def intercept_field(
        self,
    ) -> Optional[Union[Union[float, int], Iterable[Union[float, int]]]]:
        return self.yintercept

    def get_scale(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        series: Optional["pd.Series[Any]"] = None,
    ) -> Optional[Scale]:
        return view.y_scale

    def inherit_aes(self) -> bool:
        return self.inhert_aes

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.NONE,
            StatType.IDENTITY,
        ]

    def default_style(self):
        return default_line_style(self.stat_type)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.GEOM_HLINE


@dataclass
class GeomVLine(StaticLine, Geom):
    xintercept: Optional[Union[Union[float, int], Iterable[Union[float, int]]]]
    inhert_aes: bool = False

    def axis_kind(self) -> AxisKind:
        return AxisKind.X

    def get_scale(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        series: Optional["pd.Series[Any]"] = None,
    ) -> Optional[Scale]:
        return view.x_scale

    def intercept_field(
        self,
    ) -> Optional[Union[Union[float, int], Iterable[Union[float, int]]]]:
        return self.xintercept

    def inherit_aes(self) -> bool:
        return self.inhert_aes

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.NONE,
            StatType.IDENTITY,
        ]

    def default_style(self):
        return default_line_style(self.stat_type)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.GEOM_VLINE


def _ab_line(intercept: float, slope: float, x_scale: Scale, y_scale: Scale):
    intersections: List[Tuple[float, float]] = []

    if slope == 0:  # Horizontal line
        intersections.append((x_scale.low, intercept))
        intersections.append((x_scale.high, intercept))
    elif abs(slope) > 1e10:  # Nearly vertical line
        x_pos = intercept
        intersections.append((x_pos, y_scale.low))
        intersections.append((x_pos, y_scale.high))
    else:
        y_left = slope * x_scale.low + intercept
        if y_scale.low <= y_left <= y_scale.high:
            intersections.append((x_scale.low, y_left))

        y_right = slope * x_scale.high + intercept
        if y_scale.low <= y_right <= y_scale.high:
            intersections.append((x_scale.high, y_right))

        x_bottom = (y_scale.low - intercept) / slope
        if x_scale.low <= x_bottom <= x_scale.high:
            intersections.append((x_bottom, y_scale.low))

        x_top = (y_scale.high - intercept) / slope
        if x_scale.low <= x_top <= x_scale.high:
            intersections.append((x_top, y_scale.high))

    p1, p2 = intersections[:2]

    x_start = (p1[0] - x_scale.low) / (x_scale.high - x_scale.low)
    y_start = 1.0 - ((p1[1] - y_scale.low) / (y_scale.high - y_scale.low))

    x_end = (p2[0] - x_scale.low) / (x_scale.high - x_scale.low)
    y_end = 1.0 - ((p2[1] - y_scale.low) / (y_scale.high - y_scale.low))

    return x_start, y_start, x_end, y_end


@dataclass
class GeomABLine(Geom):
    intercept: Union[int, float]
    slope: Union[int, float]
    inhert_aes: bool = False

    def draw_detached_geom(
        self,
        view: ViewPort,
        filled_geom: "FilledGeom",
        style: Style,
        series: Optional["pd.Series[Any]"] = None,
    ):
        y_scale = view.y_scale
        x_scale = view.x_scale

        if y_scale is None or x_scale is None:
            raise GGException("expected a scale to draw static line")

        x_start, y_start, x_end, y_end = _ab_line(
            self.intercept, self.slope, x_scale, y_scale
        )

        start = Coord(x=RelativeCoordType(x_start), y=RelativeCoordType(y_start))
        end = Coord(x=RelativeCoordType(x_end), y=RelativeCoordType(y_end))
        line = init_line(
            start,
            end,
            InitLineInput(style=style),
        )
        view.children[0].objects.append(line)

    def inherit_aes(self) -> bool:
        return self.inhert_aes

    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.NONE,
        ]

    def default_style(self):
        return default_line_style(self.stat_type)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.GEOM_ABLINE

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


class GeomRect(GeomRectDrawMixin, Geom):
    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
        ]

    def default_style(self):
        return deepcopy(RECT_DEFAULT_STYLE)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.GEOM_RECT

    # def draw_geom(
    #     self,
    #     view: ViewPort,
    #     fg: "FilledGeom",
    #     pos: Coord,
    #     y: Any,
    #     bin_widths: Tuple[float, float],
    #     df: pd.DataFrame,
    #     idx: int,
    #     style: Style,
    # ):
    #     raise GGException("Already handled in `draw_sub_df`!")


class GeomArea(Geom):
    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.BIN,
        ]

    def default_style(self):
        return deepcopy(AREA_DEFAULT_STYLE)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.GEOM_AREA

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


class GeomLine(Geom):
    @property
    def allowed_stat_types(self) -> List["StatType"]:
        return [
            StatType.IDENTITY,
            StatType.COUNT,
            StatType.SMOOTH,
            StatType.BIN,
            StatType.DENSITY,
        ]

    def default_style(self):
        return default_line_style(self.stat_type)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.LINE

    def draw_geom(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        raise GGException("Already handled in `draw_sub_df`!")


@dataclass
class XYMinMax:
    x_min: Optional[Union[float, int]] = None
    x_max: Optional[Union[float, int]] = None
    y_min: Optional[Union[float, int]] = None
    y_max: Optional[Union[float, int]] = None

    def to_scales(
        self,
    ) -> Tuple[
        Optional["GGScale"],
        Optional["GGScale"],
        Optional["GGScale"],
        Optional["GGScale"],
    ]:
        from python_ggplot.gg.scales.base import LinearDataScale

        result = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }
        for k, v in result.items():
            if v is not None:
                result[k] = LinearDataScale.from_const(v)
        # todo change this later
        return result["x_min"], result["x_max"], result["y_min"], result["y_max"]  # type: ignore

    @classmethod
    def _evaluate_scale(
        cls,
        df: pd.DataFrame,
        scale: Optional["GGScale"],
        idx: int,
    ) -> Optional[float]:
        # move to scale
        if scale is None:
            return None
        return float(scale.evaluate(df.iloc[idx]))  # type ignore

    @classmethod
    def _min_max_for_scale(
        cls, scale: Union["GGScale", "CompositeScale"]
    ) -> Tuple[Optional["GGScale"], Optional["GGScale"]]:
        # TODO move this on the scale class
        from python_ggplot.gg.scales.base import GGScale

        if isinstance(scale, GGScale):
            return None, None
        return scale.scale_min, scale.scale_max

    @classmethod
    def from_scales(
        cls,
        df: pd.DataFrame,
        x_scale: Union["GGScale", "CompositeScale"],
        y_scale: Union["GGScale", "CompositeScale"],
        idx: int,
    ):
        xmin, xmax = cls._min_max_for_scale(x_scale)
        ymin, ymax = cls._min_max_for_scale(y_scale)
        xmin, xmax, ymin, ymax = (
            cls._evaluate_scale(df, scale, idx) for scale in [xmin, xmax, ymin, ymax]
        )
        return XYMinMax(
            x_min=xmin,
            x_max=xmax,
            y_min=ymin,
            y_max=ymax,
        )

    def _merge_value(self, left: Any, right: Any):
        if left is not None:
            return left
        return right

    def merge(self, other: "XYMinMax") -> "XYMinMax":
        """
        This is intuidive that "left is preferred"
        eg if 2 values exist left is chosen
        this will need some extension for that
        we can allow either dominant=left/right or raise exception on duplicates
        i believe the correct way is as is, but the issue is left is ambiguous
        in case of aes(xmin="col"), xmin=1 i think we are supposed to pick 1 as the value
        """
        return XYMinMax(
            x_min=self._merge_value(self.x_min, other.x_min),
            x_max=self._merge_value(self.x_max, other.x_max),
            y_min=self._merge_value(self.y_min, other.y_min),
            y_max=self._merge_value(self.y_max, other.y_max),
        )
