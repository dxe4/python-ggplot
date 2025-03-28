from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    List,
    Optional,
    OrderedDict,
    Tuple,
    cast,
)

import pandas as pd

from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import GGEnum, GGException, Scale, Style
from python_ggplot.core.units.objects import DataUnit
from python_ggplot.gg.datamancer_pandas_compat import GGValue
from python_ggplot.gg.types import (
    Aesthetics,
    BinByType,
    BinPositionType,
    DiscreteKind,
    DiscreteType,
    GGStyle,
    PositionType,
    StatBin,
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
from tests.test_view import AxisKind

if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import ColorScale


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


@dataclass
class FilledGeomData:
    geom: Geom
    x_col: Optional[str]
    y_col: Optional[str]
    x_scale: Optional[Scale]
    y_scale: Optional[Scale]
    reversed_x: bool
    reversed_y: bool
    # TODO this logic needs some reworking
    yield_data: OrderedDict[Any, Tuple[GGStyle, List[GGStyle], pd.DataFrame]]
    num_x: int
    num_y: int
    x_discrete_kind: Optional["FilledGeomDiscreteKind"]
    y_discrete_kind: Optional["FilledGeomDiscreteKind"]


@dataclass
class FilledGeom:
    """
    TODO add some of the nested data accessible here
    eg fg.gg_data.geom.geom_type -> fg.geom_type
    + rename gg_data before alpha
    """

    gg_data: FilledGeomData

    def _ensure_discrete_kind_exists(self, axis_kind: AxisKind):
        if axis_kind == AxisKind.X:
            if self.gg_data.x_discrete_kind is None:
                raise GGException("x_discrete_kind is None")
        elif axis_kind == AxisKind.Y:
            if self.gg_data.y_discrete_kind is None:
                raise GGException("y_discrete_kind is None")
        else:
            raise GGException("incorrect axis type")

    def _ensure_x_discrete_kind_exists(self):
        self._ensure_discrete_kind_exists(AxisKind.X)

    def _ensure_y_discrete_kind_exists(self):
        self._ensure_discrete_kind_exists(AxisKind.Y)

    def is_discrete_y(self) -> bool:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.discrete_type == DiscreteType.DISCRETE

    def is_discrete_x(self) -> bool:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type == DiscreteType.DISCRETE

    @property
    def discrete_type_y(self) -> DiscreteType:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.discrete_type

    @property
    def discrete_type_x(self) -> DiscreteType:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type

    @property
    def discrete_type(self) -> Optional[DiscreteType]:
        self._ensure_x_discrete_kind_exists()
        self._ensure_y_discrete_kind_exists()

        left = self.gg_data.x_discrete_kind.discrete_type
        right = self.gg_data.y_discrete_kind.discrete_type
        if left != right:
            return None
        return left

    def get_x_label_seq(self) -> List[GGValue]:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.get_label_seq()

    def get_y_label_seq(self) -> List[GGValue]:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.get_label_seq()

    @property
    def geom_type(self) -> GeomType:
        return self.gg_data.geom.geom_type

    @property
    def stat_type(self) -> StatType:
        return self.gg_data.geom.gg_data.stat_kind.stat_type

    def get_histogram_draw_style(self) -> HistogramDrawingStyle:
        # todo this will be fixed eventually just provide convinience for now
        if not self.geom_type == GeomType.HISTOGRAM:
            raise GGException("attempted to get histogram on non histogram type")
        temp = cast(FilledGeomHistogram, self)
        return temp.histogram_drawing_style

    def enumerate_data(
        self: "FilledGeom",
    ) -> Generator[Tuple[GGValue, GGStyle, List[GGStyle], pd.DataFrame], None, None]:
        for label, tup in self.gg_data.yield_data.items():
            yield label, tup[0], tup[1], tup[2]


class GeomPoint(Geom):
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

        bin_width = read_or_calc_bin_width(
            df, idx, fg.gg_data.x_col, dc_kind=fg.gg_data.x_discrete_kind.discrete_type
        )
        if bin_width != bin_widths[0]:
            raise GGException("Invalid bin width generated")

        if y is None:
            y = 0.0

        new_rect = init_rect(
            view,
            pos,
            DataUnit(bin_width),
            DataUnit(-y),
            InitRectInput(style=style, name="geom_bar_rect"),
        )
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
    @property
    def geom_type(self) -> GeomType:
        return GeomType.BAR


@dataclass
class GeomHistogram(GeomHistogramMixin, Geom):
    histogram_drawing_style: HistogramDrawingStyle

    @property
    def geom_type(self) -> GeomType:
        return GeomType.HISTOGRAM


class GeomFreqPoly(Geom):
    @property
    def geom_type(self) -> GeomType:
        return GeomType.FREQ_POLY

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

        temp = cast(FilledGeomErrorBar, fg)
        new_error_bar = draw_error_bar(view, temp, pos, df, idx, style)
        view.add_obj(new_error_bar)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.ERROR_BAR


class GeomErrorBar(GeomErrorBarMixin, Geom):
    pass


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
    pass


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
    pass


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
            DataUnit(bin_widths[1]),
            InitRectInput(style=style),
        )
        view.add_obj(new_rect)


class GeomTile(GeomTileMixin, Geom):
    pass


class GeomLine(Geom):
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


@dataclass
class FilledGeomErrorBar(GeomErrorBarMixin, FilledGeom):
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class TitleRasterData:
    fill_col: str
    fill_data_scale: Scale
    width: Optional[str]
    height: Optional[str]
    color_scale: "ColorScale"


@dataclass
class FilledGeomTitle(GeomTileMixin, FilledGeom):
    data: TitleRasterData


@dataclass
class FilledGeomRaster(GeomRasterMixin, FilledGeom):
    data: TitleRasterData


@dataclass
class FilledGeomText(GeomTextMixin, FilledGeom):
    text: str


@dataclass
class FilledGeomHistogram(GeomHistogramMixin, FilledGeom):
    histogram_drawing_style: HistogramDrawingStyle
