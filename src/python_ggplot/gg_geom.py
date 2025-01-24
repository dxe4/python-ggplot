from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, OrderedDict, Tuple, cast

import pandas as pd

from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import GGException, Scale, Style
from python_ggplot.core.units.objects import DataUnit
from python_ggplot.datamancer_pandas_compat import GGValue
from python_ggplot.gg_scales import ColorScale
from python_ggplot.gg_types import (
    Aesthetics,
    BinPositionType,
    DiscreteKind,
    DiscreteType,
    GGStyle,
    MainAddScales,
    PositionType,
    StatKind,
    StatType,
)
from python_ggplot.graphics.initialize import (
    InitRectInput,
    InitTextInput,
    init_point,
    init_rect,
    init_text,
)
from python_ggplot.graphics.views import ViewPort


class HistogramDrawingStyle(str, Enum):
    BARS = auto()
    OUTLINE = auto()


class GeomType(Enum):
    POINT = auto()
    BAR = auto()
    HISTOGRAM = auto()
    FREQ_POLY = auto()
    TILE = auto()
    LINE = auto()
    ERROR_BAR = auto()
    TEXT = auto()
    RASTER = auto()


@dataclass
class GeomKind(ABC):

    @abstractmethod
    def draw(
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


class GeomPoint(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.POINT

    def draw(
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
    def draw(
        self,
        view: ViewPort,
        fg: "FilledGeom",
        pos: Coord,
        # TODO this is Value
        y: Any,
        bin_widths: Tuple[float, float],
        df: pd.DataFrame,
        idx: int,
        style: Style,
    ):
        from python_ggplot.gg_drawing import read_or_calc_bin_width

        bin_width = read_or_calc_bin_width(
            df, idx, fg.x_col, dc_kind=fg.x_discrete_kind.discrete_type
        )
        if bin_width != bin_widths[0]:
            raise GGException("Invalid bin width generated")
        new_rect = init_rect(
            view,
            pos,
            DataUnit(bin_width),
            DataUnit(-float(y) if y is not None else 0.0),
            InitRectInput(style=style),
        )
        view.add_obj(new_rect)


class GeomBar(GeomKind, GeomRectDrawMixin):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.BAR


class GeomHistogram(GeomKind, GeomRectDrawMixin):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.HISTOGRAM


class GeomFreqPoly(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.FREQ_POLY

    def draw(
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


class GeomErrorBar(GeomKind):
    def draw(
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
        from python_ggplot.gg_drawing import draw_error_bar

        temp = cast(FilledGeomErrorBar, fg)
        new_error_bar = draw_error_bar(view, temp, pos, df, idx, style)
        view.add_obj(new_error_bar)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.ERROR_BAR


class GeomText(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.TEXT

    def draw(
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
        from python_ggplot.gg_drawing import read_text

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


class GeomRaster(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.RASTER

    def draw(
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


class GeomTile(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.TILE

    def draw(
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


class GeomLine(GeomKind):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.LINE

    def draw(
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
class Geom:
    gid: int
    kind: GeomKind
    stat_kind: StatKind
    data: Optional[pd.DataFrame] = None
    user_style: Optional[GGStyle] = None
    position: Optional[PositionType] = None
    aes: Optional[Aesthetics] = None
    bin_position: Optional[BinPositionType] = None
    # used for geom_type histogram
    histogram_drawing_style: Optional[HistogramDrawingStyle] = None

    def __post_init__(self):
        if (
            self.kind.geom_type == GeomType.HISTOGRAM
            and not self.histogram_drawing_style
        ):
            raise GGException("histogram geom needs to specify histogram_drawing_style")


class FilledGeomDiscreteKind(ABC, DiscreteKind):

    @abstractmethod
    def get_label_seq(self) -> List[GGValue]:
        pass

    @property
    @abstractmethod
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


@dataclass
class FilledGeomDiscrete(FilledGeomDiscreteKind):
    label_seq: List[GGValue]

    def get_label_seq(self) -> List[GGValue]:
        return self.label_seq

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.DISCRETE


class FilledGeomContinuous(FilledGeomDiscreteKind):
    def get_label_seq(self) -> List[GGValue]:
        raise GGException("attempt to get discrete values on continiuous kind")

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.CONTINUOUS


class FilledGeomErrorBar(GeomErrorBar):
    x_min: Optional[str]
    y_min: Optional[str]
    x_max: Optional[str]
    y_max: Optional[str]


@dataclass
class TitleRasterData:
    fill_col: str
    fill_data_scale: Scale
    width: Optional[str]
    height: Optional[str]
    color_scale: ColorScale


class FilledGeomTitle(GeomTile):
    data: TitleRasterData


class FilledGeomRaster(GeomRaster):
    data: TitleRasterData


class FilledGeomText(GeomText):
    text: str


class FilledGeomHistogram(GeomHistogram):
    histogram_drawing_style: HistogramDrawingStyle


@dataclass
class FilledGeom:
    """
    todo, spend some time thinking on this
    the original code is using an enum for geom_type
    to decide what attributes are defined on the class
    we have many choices here but maybe we can settle with something like this?
    FilledGeom(geom=Geom(kind=FilledGeomHistogram))
    we could also use FilledGeom(geom=Geom(), kind=FilledGeomHistogram)
    the second method allows us to define a union of the types,
    so we make sure we dont end up with the wrong types accidentally
    """

    geom: Geom
    x_col: str
    y_col: str
    x_scale: Scale
    y_scale: Scale
    reversed_x: bool
    reversed_y: bool
    yield_data: OrderedDict[GGValue, Tuple[GGStyle, List[GGStyle], pd.DataFrame]]
    num_x: int
    num_y: int
    x_discrete_kind: FilledGeomDiscreteKind
    y_discrete_kind: FilledGeomDiscreteKind

    def is_discrete_y(self) -> bool:
        return self.y_discrete_kind.discrete_type == DiscreteType.DISCRETE

    def is_discrete_x(self) -> bool:
        return self.x_discrete_kind.discrete_type == DiscreteType.DISCRETE

    @property
    def discrete_type_y(self) -> DiscreteType:
        return self.y_discrete_kind.discrete_type

    @property
    def discrete_type_x(self) -> DiscreteType:
        return self.x_discrete_kind.discrete_type

    @property
    def discrete_type(self) -> Optional[DiscreteType]:
        left = self.x_discrete_kind.discrete_type
        right = self.y_discrete_kind.discrete_type
        if left != right:
            return None
        return left

    def get_x_label_seq(self) -> List[GGValue]:
        return self.x_discrete_kind.get_label_seq()

    def get_y_label_seq(self) -> List[GGValue]:
        return self.y_discrete_kind.get_label_seq()

    @property
    def geom_type(self) -> GeomType:
        return self.geom.kind.geom_type

    @property
    def stat_type(self) -> StatType:
        return self.geom.stat_kind.stat_type

    def get_histogram_draw_style(self) -> HistogramDrawingStyle:
        # todo this will be fixed eventually just provide convinience for now
        if not self.geom_type == GeomType.HISTOGRAM:
            raise GGException("attempted to get histogram on non histogram type")
        temp = cast(FilledGeomHistogram, self)
        return temp.histogram_drawing_style


@dataclass
class FilledScales:
    x_scale: Scale
    y_scale: Scale
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
    facets: List[Scale]
