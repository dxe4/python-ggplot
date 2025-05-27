from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import auto
from math import isclose
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
    cast,
)

import pandas as pd

from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import GGEnum, GGException, Scale, Style
from python_ggplot.core.units.objects import DataUnit
from python_ggplot.gg.datamancer_pandas_compat import GGValue, VectorCol, VNull
from python_ggplot.gg.styles.config import (
    AREA_DEFAULT_STYLE,
    BAR_DEFAULT_STYLE,
    HISTO_DEFAULT_STYLE,
    LINE_DEFAULT_STYLE,
    POINT_DEFAULT_STYLE,
    SMOOTH_DEFAULT_STYLE,
    TEXT_DEFAULT_STYLE,
    TILE_DEFAULT_STYLE,
)
from python_ggplot.gg.types import (
    COUNT_COL,
    SMOOTH_VALS_COL,
    Aesthetics,
    BinByType,
    BinPositionType,
    ColOperator,
    DiscreteKind,
    DiscreteType,
    GgPlot,
    GGStyle,
    PositionType,
    StatBin,
    StatKind,
    StatType,
)
from python_ggplot.graphics.initialize import (
    InitRectInput,
    InitTextInput,
    calc_tick_locations,
    init_point,
    init_rect,
    init_text,
)
from python_ggplot.graphics.views import ViewPort
from tests.test_view import AxisKind

if TYPE_CHECKING:
    from python_ggplot.gg.scales.base import (
        ColorScale,
        FilledScales,
        GGScale,
        MainAddScales,
    )


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


def default_line_style(stat_type: StatType):
    if stat_type == StatType.SMOOTH:
        return deepcopy(SMOOTH_DEFAULT_STYLE)
    else:
        return deepcopy(LINE_DEFAULT_STYLE)


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

    x_transformations: Optional[List[ColOperator]] = None
    y_transformations: Optional[List[ColOperator]] = None

    def is_x_discrete(self):
        return isinstance(self.x_discrete_kind, FilledGeomDiscrete)

    def is_y_discrete(self):
        return isinstance(self.y_discrete_kind, FilledGeomDiscrete)

    def is_x_continuous(self):
        return isinstance(self.x_discrete_kind, FilledGeomContinuous)

    def is_y_continuous(self):
        return isinstance(self.y_discrete_kind, FilledGeomContinuous)


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
        return self.gg_data.y_discrete_kind.discrete_type == DiscreteType.DISCRETE  # type: ignore

    def is_discrete_x(self) -> bool:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type == DiscreteType.DISCRETE  # type: ignore

    @property
    def discrete_type_y(self) -> DiscreteType:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.discrete_type  # type: ignore

    @property
    def discrete_type_x(self) -> DiscreteType:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.discrete_type  # type: ignore

    @property
    def discrete_type(self) -> Optional[DiscreteType]:
        self._ensure_x_discrete_kind_exists()
        self._ensure_y_discrete_kind_exists()

        left = self.gg_data.x_discrete_kind.discrete_type  # type: ignore
        right = self.gg_data.y_discrete_kind.discrete_type  # type: ignore
        if left != right:
            return None
        return left

    def get_x_label_seq(self) -> List[GGValue]:
        self._ensure_x_discrete_kind_exists()
        return self.gg_data.x_discrete_kind.get_label_seq()  # type: ignore

    def get_y_label_seq(self) -> List[GGValue]:
        self._ensure_y_discrete_kind_exists()
        return self.gg_data.y_discrete_kind.get_label_seq()  # type: ignore

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

    def maybe_filter_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        # this is only needed for FilledGeomErrorBar as of now
        return df


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

        temp_fg = cast(FilledGeomErrorBar, fg)
        new_error_bar = draw_error_bar(view, temp_fg, pos, df, idx, style)
        view.add_obj(new_error_bar)

    @property
    def geom_type(self) -> GeomType:
        return GeomType.ERROR_BAR


class GeomErrorBar(GeomErrorBarMixin, Geom):

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


@dataclass
class FilledGeomContinuous(FilledGeomDiscreteKind):
    def get_label_seq(self) -> List[GGValue]:
        raise GGException("attempt to get discrete values on continiuous kind")

    @property
    def discrete_type(self) -> DiscreteType:
        return DiscreteType.CONTINUOUS


def _optional_scale_col(scale: Optional["GGScale"]) -> Optional[str]:
    if scale is None:
        return None
    return scale.get_col_name()


@dataclass
class FilledGeomErrorBar(GeomErrorBarMixin, FilledGeom):
    x_min: Optional[str] = None
    y_min: Optional[str] = None
    x_max: Optional[str] = None
    y_max: Optional[str] = None

    def maybe_filter_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        x_values = [i for i in [self.x_min, self.x_max] if i is not None]
        y_values = [i for i in [self.y_min, self.y_max] if i is not None]
        collect_cols = x_values + y_values

        if len(x_values) > 0:
            if self.gg_data.y_col is None:
                raise GGException("expected y_col")
            collect_cols.append(self.gg_data.y_col)

        if len(y_values) > 0:
            if self.gg_data.x_col is None:
                raise GGException("expected x_col")
            collect_cols.append(self.gg_data.x_col)

        return df.drop_duplicates(subset=collect_cols)

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomErrorBar(
            gg_data=fg_data,
            x_min=_optional_scale_col(fs.get_x_min_scale(geom, optional=True)),
            x_max=_optional_scale_col(fs.get_x_max_scale(geom, optional=True)),
            y_min=_optional_scale_col(fs.get_y_min_scale(geom, optional=True)),
            y_max=_optional_scale_col(fs.get_y_max_scale(geom, optional=True)),
        )
        return new_fg, df


@dataclass
class TitleRasterData:
    fill_col: str
    fill_data_scale: Optional[Scale]
    width: Optional[str]
    height: Optional[str]
    color_scale: Optional["ColorScale"]

    @staticmethod
    def _get_width(
        geom: Geom, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[str, pd.DataFrame]:
        # todo clean up
        width_scale = fs.get_width_scale(geom, optional=True)
        x_min_s = fs.get_x_min_scale(geom, optional=True)
        x_max_s = fs.get_x_max_scale(geom, optional=True)
        # Handle width
        if width_scale is not None:
            width = width_scale.get_col_name()  # type: ignore
            return width, df

        elif x_min_s is not None and x_max_s is not None:
            min_name = x_min_s.get_col_name()  # type: ignore
            max_name = x_max_s.get_col_name()  # type: ignore
            x_col_name = get_x_scale(fs, fg.geom).get_col_name()  # type: ignore
            df["width"] = df[max_name] - df[min_name]
            df[x_col_name] = df[min_name]
            return "width", df
        elif x_min_s is not None or x_max_s is not None:
            raise GGException(
                "Invalid combination of aesthetics! If no width given both an `x_min` and `x_max` has to be supplied for geom_{fg.geom_kind}!"
            )
        else:
            if geom.geom_type == GeomType.RASTER:
                x_col = df[get_x_scale(fs, fg.geom).get_col_name()].unique()  # type: ignore
                fg.num_x = len(x_col)  # type: ignore
                df["width"] = abs(x_col[1] - x_col[0])  # type: ignore
            else:
                print(
                    "INFO: using default width of 1 since no width information supplied. "
                    "Add `width` or (`x_min`, `x_max`) as aesthetics for different values."
                )
                df["width"] = 1.0
            return "width", df

    @staticmethod
    def _get_height(
        geom: Geom, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[str, pd.DataFrame]:
        # todo clean up
        height_scale = fs.get_height_scale(geom, optional=True)
        y_min_s = fs.get_y_min_scale(geom, optional=True)
        y_max_s = fs.get_y_max_scale(geom, optional=True)

        if height_scale is not None:
            return height_scale.get_col_name(), df

        elif y_min_s is not None and y_max_s is not None:
            min_name = y_min_s.get_col_name()
            max_name = y_max_s.get_col_name()

            y_scale = fs.get_y_scale(geom)
            if y_scale is None:
                raise GGException("expected a y_scale")

            y_col_name = y_scale.get_col_name()
            df["height"] = df[max_name] - df[min_name]
            df[y_col_name] = df[min_name]
            return "height", df

        elif y_min_s is not None or y_max_s is not None:
            raise GGException(
                "Invalid combination of aesthetics! If no height given both an `y_min` and `y_max` has to be supplied for geom_{fg.geom_kind}!"
            )
        else:
            if geom.geom_type == GeomType.RASTER:
                col_name = get_y_scale(fs, fg.geom).get_col_name()  # type: ignore
                y_col = df[col_name].unique()  # type: ignore
                fg.num_y = len(y_col)  # type: ignore
                df["height"] = abs(y_col[1] - y_col[0])  # type: ignore
            else:
                print(
                    "INFO: using default height of 1 since no height information supplied. "
                    "Add `height` or (`y_min`, `y_max`) as aesthetics for different values."
                )
                df["height"] = 1.0
            return "height", df

    @staticmethod
    def get_height_and_width(
        geom: Geom, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[str], pd.DataFrame]:

        height, df = TitleRasterData._get_height(geom, fs, df)
        width, df = TitleRasterData._get_width(geom, fs, df)

        return width, height, df


def create_filled_geom_tile_and_raster(
    cls: Union[Type["FilledGeomTitle"], Type["FilledGeomRaster"]],
    geom: Geom,
    fg_data: FilledGeomData,
    fs: "FilledScales",
    df: pd.DataFrame,
) -> Tuple[FilledGeom, pd.DataFrame]:
    from python_ggplot.gg.styles.utils import use_or_default

    fill_data_scale: Optional[Scale] = None
    color_scale: Optional["ColorScale"] = None
    fill_col: str = ""

    width, height, df = TitleRasterData.get_height_and_width(geom, fs, df)

    fill_scale = fs.get_fill_scale(geom)
    if fill_scale is None:
        raise GGException("requires a `fill` aesthetic scale!")

    fill_col = fill_scale.get_col_name()
    if fill_scale.is_continuous():
        fill_data_scale = fill_scale.gg_data.discrete_kind.data_scale  # type: ignore
        # TODO fix this, fine for now
        potential_color_scale: Optional[ColorScale] = getattr(
            fill_scale, "color_scale", None
        )
        color_scale = use_or_default(potential_color_scale)

    tile_raster_data = TitleRasterData(
        fill_col=fill_col,
        fill_data_scale=fill_data_scale,  # type: ignore
        width=width,
        height=height,
        color_scale=color_scale,
    )
    new_fg = cls(gg_data=fg_data, data=tile_raster_data)
    return new_fg, df


@dataclass
class FilledGeomTitle(GeomTileMixin, FilledGeom):
    data: TitleRasterData

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        return create_filled_geom_tile_and_raster(cls, geom, fg_data, fs, df)


@dataclass
class FilledGeomRaster(GeomRasterMixin, FilledGeom):
    data: TitleRasterData

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        return create_filled_geom_tile_and_raster(cls, geom, fg_data, fs, df)


@dataclass
class FilledGeomText(GeomTextMixin, FilledGeom):
    text: str

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomText(
            gg_data=fg_data,
            text=str(fs.get_text_scale(geom).gg_data.col),  # type: ignore
        )
        return new_fg, df


@dataclass
class FilledGeomHistogram(GeomHistogramMixin, FilledGeom):
    histogram_drawing_style: HistogramDrawingStyle

    @classmethod
    def from_geom(
        cls, geom: Geom, fg_data: FilledGeomData, fs: "FilledScales", df: pd.DataFrame
    ) -> Tuple[FilledGeom, pd.DataFrame]:
        new_fg = FilledGeomHistogram(
            gg_data=fg_data,
            histogram_drawing_style=geom.histogram_drawing_style,  # type: ignore
        )
        return new_fg, df


def create_filled_geom(
    fg: FilledGeom, fs: "FilledScales", geom_type: GeomType, df: pd.DataFrame
) -> Tuple[FilledGeom, pd.DataFrame]:
    # Originally "fill_opt_fields"
    # this is a bit ugly, but its lot better than the original
    # its readable enough for now
    if fg.geom_type == GeomType.ERROR_BAR:
        return FilledGeomErrorBar.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    # tile and raster is really the same. fine for now
    elif fg.geom_type == GeomType.TILE:
        return FilledGeomTitle.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.RASTER:
        return FilledGeomRaster.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.TEXT:
        return FilledGeomText.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    elif fg.geom_type == GeomType.HISTOGRAM:
        return FilledGeomHistogram.from_geom(fg.gg_data.geom, fg.gg_data, fs, df)
    else:
        return fg, df


def apply_transformations(df: pd.DataFrame, scales: List["GGScale"]):
    """
    TODO this will need fixing
    """
    from python_ggplot.gg.scales.base import ScaleType

    transformations: Dict[Any, Any] = {}
    result: pd.DataFrame = pd.DataFrame()

    for scale in scales:
        if scale.scale_type == ScaleType.TRANSFORMED_DATA:
            # TODO formula node logic should be wrong
            col = scale.col.evaluate(df)  # type: ignore
            # This is probably wrong too
            col_str = scale.get_col_name()

            transformations[col_str] = lambda x, s=scale, c=col: s.trans(df[c])  # type: ignore
        else:
            # TODO this can only be VectorCol for now i think
            # but this is FomrulaNode logic, which we may add in the near future
            col = scale.gg_data.col
            if isinstance(col, str):
                transformations[col] = lambda x, c: x[c]  # type: ignore
            elif isinstance(col, VectorCol):  # type: ignore
                transformations[col.col_name] = lambda x, c: x[c]  # type: ignore
            else:
                # Assume col is some kind of formula/expression that can be evaluated
                transformations[scale.get_col_name()] = lambda x, c: c.evaluate()  # type: ignore

    for col_name, transform_fn in transformations.items():
        result[col_name] = transform_fn(df, col_name)
    return result


def post_process_scales(filled_scales: "FilledScales", plot: "GgPlot"):
    # keeping as is for backwards compatibility for now
    create_filled_geoms_for_filled_scales(filled_scales, plot)


def split_discrete_set_map(
    df: pd.DataFrame, scales: List["GGScale"]  # type: ignore
) -> Tuple[List[str], List[str]]:
    set_disc_cols: List[str] = []
    map_disc_cols: List[str] = []

    for scale in scales:
        if str(scale.gg_data.col) in df.columns:
            if str(scale.gg_data.col) not in map_disc_cols:
                map_disc_cols.append(str(scale.gg_data.col))
        else:
            if str(scale.gg_data.col) not in set_disc_cols:
                set_disc_cols.append(str(scale.gg_data.col))

    return set_disc_cols, map_disc_cols


def get_scales(
    geom: Geom, filled_scales: "FilledScales", y_is_none: bool = False
) -> Tuple[Optional["GGScale"], Optional["GGScale"], List["GGScale"]]:
    gid = geom.gg_data.gid

    def get_scale(field: Optional["MainAddScales"]) -> Optional["GGScale"]:
        if field is None:
            # TODO is this exception correct?
            raise GGException("attempted to get on empty scale")
        more_scale = [s for s in field.more or [] if gid in s.gg_data.ids]
        if len(more_scale) > 1:
            raise GGException("found more than 1 scale matching gid")
        if len(more_scale) == 1:
            return more_scale[0]
        elif field.main is not None:
            return field.main
        else:
            return None

    x_opt = get_scale(filled_scales.x)
    y_opt = get_scale(filled_scales.y)

    if y_is_none and y_opt is not None and x_opt is None:
        # TODO high priority
        # if only y is given, we flip the plot
        # this really shouldnt happen, the previous behaviour was that both x and y have to be given
        # so this is a step forward
        # some geom logic is hard coded so that x is the default
        # there is a plan to refactor this soon
        y_opt, x_opt = x_opt, y_opt

    other_scales: List["GGScale"] = []

    attrs_ = [
        filled_scales.color,
        filled_scales.fill,
        filled_scales.size,
        filled_scales.shape,
        filled_scales.x_min,
        filled_scales.x_max,
        filled_scales.y_min,
        filled_scales.y_max,
        filled_scales.width,
        filled_scales.height,
        filled_scales.text,
        filled_scales.y_ridges,
        filled_scales.width,
    ]

    for attr_ in attrs_:
        new_scale = get_scale(attr_)
        if new_scale is not None:
            other_scales.append(new_scale)

    other_scales.extend(filled_scales.facets)
    return x_opt, y_opt, other_scales


def separate_scales_apply_transofrmations(
    df: pd.DataFrame,  # type: ignore
    geom: Geom,
    filled_scales: "FilledScales",
    y_is_none: bool = False,
) -> Tuple[Optional["GGScale"], Optional["GGScale"], List["GGScale"], List["GGScale"]]:
    """
    TODO test this
    """
    x, y, scales = get_scales(geom, filled_scales, y_is_none=y_is_none)

    discretes = [s for s in scales if s.is_discrete()]
    cont = [s for s in scales if s.is_continuous()]

    discr_cols = list(
        set(s.get_col_name() for s in discretes if s.get_col_name() in df.columns)
    )

    if len(discr_cols) > 0:
        df = df.groupby(discr_cols, group_keys=True)  # type: ignore

    # TODO urgent double check this
    # We may not need this until FormulaNode is Implemented

    # if not y_is_none:
    #     apply_transformations(df, [x, y] + scales)  # type: ignore
    # else:
    #     apply_transformations(df, [x] + scales)

    return (x, y, discretes, cont)


def encompassing_data_scale(
    scales: List["GGScale"],
    axis_kind: AxisKind,
    base_scale: tuple[float, float] = (0.0, 0.0),
) -> Scale:
    from python_ggplot.gg.scales.base import (
        GGScaleContinuous,
        LinearDataScale,
        TransformedDataScale,
    )

    result = Scale(low=base_scale[0], high=base_scale[1])

    for scale_ in scales:
        if isinstance(scale_, (LinearDataScale, TransformedDataScale)):
            if scale_.data is not None and scale_.data.axis_kind == axis_kind:
                if isinstance(scale_.gg_data.discrete_kind, GGScaleContinuous):
                    # TODO double check, why does original code not check for continuous?
                    result = result.merge(scale_.gg_data.discrete_kind.data_scale)

    return result


def _get_scale_col_name(scale: Optional["GGScale"]) -> Optional[str]:
    if scale is None:
        return None
    return scale.get_col_name()


def _get_filled_geom_from_scale(scale: Optional["GGScale"]):
    # todo rename
    if scale is None:
        return None
    return scale.gg_data.discrete_kind.to_filled_geom_kind()


def determine_data_scale(
    scale: Optional["GGScale"], additional: List["GGScale"], df: pd.DataFrame
) -> Optional[Scale]:
    from python_ggplot.gg.scales.base import GGScaleContinuous, GGScaleDiscrete

    if scale is None:
        return None

    if not str(scale.gg_data.col) in df.columns:
        # TODO, port this logic on formula node
        raise GGException("col not in df")

    if isinstance(scale.gg_data.discrete_kind, GGScaleContinuous):
        # happens for input DFs with 1-2 elements
        existing_scale = scale.gg_data.discrete_kind.data_scale
        if existing_scale is not None:
            return existing_scale
        else:
            low, high = (df[str(scale.gg_data.col)].min(), df[str(scale.col)].max())  # type: ignore
            return Scale(low=low, high=high)  # type: ignore
    elif isinstance(scale.gg_data.discrete_kind, GGScaleDiscrete):
        # for discrete case assign default [0, 1] scale
        return Scale(low=0.0, high=1.0)
    else:
        raise GGException("unexpected discrete kind")


def stat_kind_fg_class(stat_type: StatType) -> Type["FilledStatGeom"]:
    lookup = {
        StatType.IDENTITY: FilledIdentityGeom,
        StatType.COUNT: FilledCountGeom,
        StatType.SMOOTH: FilledSmoothGeom,
        StatType.BIN: FilledBinGeom,
        StatType.DENSITY: FilledBinGeom,
    }
    if stat_type not in lookup:
        raise GGException(f"unsuppoerted stat type {stat_type}")

    return lookup[stat_type]


def create_fillsed_scale_stat_geom(
    df: pd.DataFrame, geom: Any, filled_scales: "FilledScales"
) -> "FilledStatGeom":
    x, y, discrete_scales, continuous_scales = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discrete_scales)
    filled_stat_geom_cls = stat_kind_fg_class(geom.stat_type)

    fsg = filled_stat_geom_cls(
        geom=geom,
        df=df,
        x=x,
        y=y,
        discrete_scales=discrete_scales,
        continuous_scales=continuous_scales,
        set_discrete_columns=set_disc_cols,
        map_discrete_columns=map_disc_cols,
    )
    return fsg


def create_filled_geom_from_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: "FilledScales"
) -> "FilledGeom":
    if geom.stat_type not in geom.allowed_stat_types:
        raise GGException(
            f"{geom} has stat_type {geom.stat_type} but onle allowed {geom.allowed_stat_types}"
        )

    filled_scale_stat_geom = create_fillsed_scale_stat_geom(df, geom, filled_scales)
    filled_geom, df, stlye = filled_scale_stat_geom.create_filled_geom(filled_scales)
    filled_scale_stat_geom.post_process(filled_geom, df)
    return filled_geom


def excpand_scale(scale: Scale, is_continuous: bool):
    """
    an implementation of https://ggplot2.tidyverse.org/reference/expansion.html
    this needs to be configurable, but by default there's an expansion,
    so we add the default one
    need to check how the original ggplot does this
    what is implemented is "fine/good enough for now"
    but for some cases it makes the plots a bit ugly
    maybe the ideal scenario is to expand only if there's elements that go out of the plot
    for example test_geom_linerange
    """

    if not is_continuous:
        return scale

    if scale.low == 0.0:
        return scale

    diff = scale.high - scale.low
    space = diff * 0.1
    scale.low = scale.low - space
    scale.high = scale.high + space

    return scale


def create_filled_geoms_for_filled_scales(
    filled_scales: "FilledScales", plot: "GgPlot"
):
    from python_ggplot.gg.ticks import get_x_ticks, get_y_ticks

    x_scale: Optional[Scale] = None
    y_scale: Optional[Scale] = None

    x_continuous = False
    y_continuous = False

    for geom in plot.geoms:
        if geom.gg_data.data is not None:
            df = geom.gg_data.data
        else:
            df = plot.data.copy(deep=False)

        filled_geom = create_filled_geom_from_geom(df, geom, filled_scales)

        x_continuous = x_continuous or filled_geom.gg_data.is_x_continuous()
        y_continuous = y_continuous or filled_geom.gg_data.is_y_continuous()

        if (
            x_scale is not None
            and not x_scale.is_empty()
            and y_scale is not None
            and not y_scale.is_empty()
        ):
            x_scale = x_scale.merge(filled_geom.gg_data.x_scale)
            y_scale = y_scale.merge(filled_geom.gg_data.y_scale)
        else:
            x_scale = filled_geom.gg_data.x_scale
            y_scale = filled_geom.gg_data.y_scale

        filled_scales.geoms.append(filled_geom)

    if x_scale is None or y_scale is None:
        raise GGException("x and y scale have not exist by this point")

    final_x_scale, _, _ = calc_tick_locations(x_scale, get_x_ticks(filled_scales))
    final_y_scale, _, _ = calc_tick_locations(y_scale, get_y_ticks(filled_scales))

    final_x_scale = excpand_scale(final_x_scale, x_continuous)
    final_y_scale = excpand_scale(final_y_scale, y_continuous)

    filled_scales.x_scale = final_x_scale
    filled_scales.y_scale = final_y_scale


@dataclass
class FilledStatGeom(ABC):
    geom: Geom
    df: pd.DataFrame
    x: Optional["GGScale"]
    y: Optional["GGScale"]
    discrete_scales: List["GGScale"]
    continuous_scales: List["GGScale"]
    set_discrete_columns: List["str"]
    map_discrete_columns: List["str"]

    @abstractmethod
    def fill_crated_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        pass

    def _get_col_transformations(
        self, scale: Optional["GGScale"]
    ) -> Optional[List[ColOperator]]:
        if scale is None:
            return None
        return scale.gg_data.col.get_transformations()

    def create_filled_geom(
        self, filled_scales: "FilledScales"
    ) -> Tuple[FilledGeom, pd.DataFrame, "GGStyle"]:
        from python_ggplot.gg.styles.utils import apply_style

        self.validate()
        fg_data = FilledGeomData(
            geom=self.geom,
            x_col=self.get_x_col(),
            y_col=self.get_y_col(),
            x_scale=self.get_x_scale(),
            y_scale=self.get_y_scale(),
            reversed_x=False,
            reversed_y=False,
            yield_data={},  # type: ignore
            x_discrete_kind=self.get_x_discrete_kind(),
            y_discrete_kind=self.get_y_discrete_kind(),
            num_x=0,
            num_y=0,
            x_transformations=self._get_col_transformations(self.x),
            y_transformations=self._get_col_transformations(self.y),
        )
        fg = FilledGeom(gg_data=fg_data)
        fg, df = create_filled_geom(fg, filled_scales, self.geom.geom_type, self.df)

        style = GGStyle()
        apply_style(
            style,
            df,
            self.discrete_scales,
            [(col, VNull()) for col in self.set_discrete_columns],
        )
        self.geom.gg_data.data = df
        self.df = df
        fg = self.fill_crated_geom(filled_scales, fg, style)
        return fg, df, style

    @abstractmethod
    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def get_x_col(self) -> Optional[str]:
        pass

    @abstractmethod
    def get_y_col(self) -> Optional[str]:
        pass

    @abstractmethod
    def get_x_scale(self) -> Optional["Scale"]:
        pass

    @abstractmethod
    def get_y_scale(self) -> Optional["Scale"]:
        pass

    @abstractmethod
    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass

    @abstractmethod
    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        pass


class FilledSmoothGeom(FilledStatGeom):
    def fill_crated_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_smooth_geom

        return filled_smooth_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_discrete():
            raise GGException("expected continuous data")

        if self.y is not None and self.y.is_discrete():
            raise GGException("expected continuous data")

        if self.y is None:
            # TODO i think this logic is wrong, double check
            raise GGException("y is none")

    def get_x_col(self) -> Optional[str]:
        return self.x.get_col_name()

    def get_y_col(self) -> Optional[str]:
        return SMOOTH_VALS_COL

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.y, self.continuous_scales, self.df)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledBinGeom(FilledStatGeom):
    def fill_crated_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_bin_geom

        return filled_bin_geom(self.df, filled_geom, self, filled_scales, style)

    def count_col(self):
        stat_kind = self.geom.gg_data.stat_kind
        # TODO double check if this was the intention, but i think it is
        if getattr(stat_kind, "density", False):
            return "density"
        else:
            return COUNT_COL

    def width_col(self):
        return "binWidths"

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_discrete():
            raise GGException("For discrete data columns use `geom_bar` instead!")

    def get_x_col(self) -> Optional[str]:
        return self.x.get_col_name()

    def get_y_col(self) -> Optional[str]:
        return self.count_col()

    def get_x_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.X)

    def get_y_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.Y)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledCountGeom(FilledStatGeom):
    def fill_crated_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_count_geom

        return filled_count_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        pass

    def validate(self):
        if self.x.is_continuous():
            raise GGException(
                "For continuous data columns use `geom_histogram` instead!"
            )

    def get_x_col(self) -> Optional[str]:
        return self.x.get_col_name()

    def get_y_col(self) -> Optional[str]:
        return COUNT_COL

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return encompassing_data_scale(self.continuous_scales, AxisKind.Y)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        if self.x.is_discrete():
            # TODO critical, easy task
            # double check if we need to pass empty label_seq
            # or if we need x.gg_data.discrete_kind.label_seq
            return FilledGeomDiscrete(label_seq=[])
        else:
            return FilledGeomContinuous()

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return FilledGeomContinuous()


class FilledIdentityGeom(FilledStatGeom):
    def fill_crated_geom(
        self, filled_scales: "FilledScales", filled_geom: "FilledGeom", style: "GGStyle"
    ) -> "FilledGeom":
        from python_ggplot.gg.geom.utils import filled_identity_geom

        return filled_identity_geom(self.df, filled_geom, self, filled_scales, style)

    def post_process(self, fg: FilledGeom, df: pd.DataFrame):
        if self.geom.geom_type not in {
            GeomType.HISTOGRAM,
            GeomType.FREQ_POLY,
            GeomType.BAR,
        }:
            return

        if fg.gg_data.y_scale is None:
            raise GGException(
                f"expected y scale for geom {self.geom.geom_type} and stat_type IDENTIDY"
            )

        fg.gg_data.y_scale = Scale(
            low=min(0.0, fg.gg_data.y_scale.low),
            high=fg.gg_data.y_scale.high,
        )

    def validate(self):
        pass

    def get_x_col(self) -> Optional[str]:
        return _get_scale_col_name(self.x)

    def get_y_col(self) -> Optional[str]:
        return _get_scale_col_name(self.y)

    def get_x_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.x, self.continuous_scales, self.df)

    def get_y_scale(self) -> Optional["Scale"]:
        return determine_data_scale(self.y, self.continuous_scales, self.df)

    def get_x_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return _get_filled_geom_from_scale(self.x)

    def get_y_discrete_kind(self) -> Optional["FilledGeomDiscreteKind"]:
        return _get_filled_geom_from_scale(self.y)
