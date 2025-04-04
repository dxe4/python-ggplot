from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import auto
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

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.common.maths import histogram
from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import GGEnum, GGException, Scale, Style
from python_ggplot.core.units.objects import DataUnit
from python_ggplot.gg.datamancer_pandas_compat import GGValue, VNull, VString
from python_ggplot.gg.types import (
    COUNT_COL,
    PREV_VALS_COL,
    SMOOTH_VALS_COL,
    Aesthetics,
    BinByType,
    BinPositionType,
    DiscreteKind,
    DiscreteType,
    GgPlot,
    GGStyle,
    PositionType,
    SmoothMethodType,
    StatBin,
    StatKind,
    StatSmooth,
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

    @abstractmethod
    def create_filled_geom(self, df: pd.DataFrame, filled_scales: "FilledScales"):
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


class XYCreateFilledGeom:

    def create_filled_geom(self, df: pd.DataFrame, filled_scales: "FilledScales"):
        pass


class HistFreqPolyCreateFilledGeom:

    def create_filled_geom(self, df: pd.DataFrame, filled_scales: "FilledScales"):
        pass


class GeomPoint(XYCreateFilledGeom, Geom):
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


class GeomHistogramMixin(HistFreqPolyCreateFilledGeom, GeomRectDrawMixin):
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

    def create_filled_geom(self, df: pd.DataFrame, filled_scales: "FilledScales"):
        pass


@dataclass
class GeomHistogram(GeomHistogramMixin, Geom):
    histogram_drawing_style: HistogramDrawingStyle

    @property
    def geom_type(self) -> GeomType:
        return GeomType.HISTOGRAM


class GeomFreqPoly(HistFreqPolyCreateFilledGeom, Geom):
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


class GeomErrorBar(XYCreateFilledGeom, GeomErrorBarMixin, Geom):
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


class GeomText(XYCreateFilledGeom, GeomTextMixin, Geom):
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


class GeomRaster(XYCreateFilledGeom, GeomRasterMixin, Geom):
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


class GeomTile(XYCreateFilledGeom, GeomTileMixin, Geom):
    pass


class GeomLine(XYCreateFilledGeom, Geom):
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
        width_scale = fs.get_width_scale(geom)
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
        height_scale = fs.get_height_scale(geom)
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


def apply_cont_scale_if_any(
    yield_df: pd.DataFrame,
    scales: List["GGScale"],
    base_style: GGStyle,
    geom_type: GeomType,
    to_clone: bool = False,
):
    from python_ggplot.gg.scales.base import ScaleType
    from python_ggplot.gg.styles.utils import change_style

    result_style = base_style
    result_styles = []
    result_df = yield_df.copy() if to_clone else yield_df

    for scale in scales:
        # TODO col eval is a global issue, fine for now
        result_df[scale.get_col_name()] = scale.gg_data.col.evaluate(result_df)  # type: ignore

        if scale.scale_type in {ScaleType.TRANSFORMED_DATA, ScaleType.LINEAR_DATA}:
            pass
        else:
            # avoid expensive computation for raster
            if geom_type != GeomType.RASTER:
                # TODO high priority map_data logic is funny overall, add ignore type for now
                sc_vals = scale.map_data(result_df)
                result_styles = [change_style(base_style, val) for val in sc_vals]

    if not result_styles:
        result_styles = [base_style]

    return (result_style, result_styles, result_df)


def add_zero_keys(
    df: pd.DataFrame, keys: pd.Series, x_col: Any, count_col: str
) -> pd.DataFrame:
    exist_keys = df[x_col].unique()  # type: ignore
    df_zero = pd.DataFrame({x_col: keys[~keys.isin(exist_keys)]})  # type: ignore
    df_zero[count_col] = 0
    return pd.concat([df, df_zero], ignore_index=True)


def _scale_to_numpy_array(df: pd.DataFrame, scale: Optional["GGScale"]) -> NDArray[np.floating[Any]]:
    if scale is None:
        return np.empty(0, dtype=np.float64)
    else:
        return df[str(scale.gg_data.col)].to_numpy(dtype=float)  # type: ignore


def call_hist(
    df: pd.DataFrame,
    bins_arg: Any,
    stat_kind: StatBin,
    range_scale: Scale,
    weight_scale: Optional["GGScale"],
    data: NDArray[np.floating[Any]],
):
    if stat_kind.bin_by == BinByType.FULL:
        range_val = (range_scale.low, range_scale.high)
    else:
        range_val = (0.0, 0.0)

    weight_data = _scale_to_numpy_array(df, weight_scale)
    if len(weight_data) == 0:
        weight_data = None

    hist, bin_edges = histogram(
        data,
        bins_arg,
        weights=weight_data,
        range=range_val,
        density=stat_kind.density,
    )
    return hist, bin_edges


def call_histogram(
    geom: Geom,
    df: pd.DataFrame,
    scale: Optional["GGScale"],
    weight_scale: Optional["GGScale"],
    range_scale: Scale,
) -> Tuple[
    List[float],
    List[float],
    List[float],
]:
    """
    TODO revisti this once public interface is ready
    """
    stat_kind = geom.gg_data.stat_kind
    if not isinstance(stat_kind, StatBin):
        raise GGException("expected bin stat type")

    data = _scale_to_numpy_array(df, scale)
    hist = []
    bin_edges = []
    bin_widths = []

    if stat_kind.bin_edges is not None:
        hist, bin_edges = call_hist(
            df, stat_kind.bin_edges, stat_kind, range_scale, weight_scale, data
        )
    elif stat_kind.bin_width is not None:
        bins = round((range_scale.high - range_scale.low) / stat_kind.bin_width)
        hist, bin_edges = call_hist(
            df, int(bins), stat_kind, range_scale, weight_scale, data
        )
    else:
        hist, bin_edges = call_hist(
            df, stat_kind.num_bins, stat_kind, range_scale, weight_scale, data
        )

    bin_widths = np.diff(bin_edges)  # type: ignore
    # TODO CRITICAL+ sanity this logic
    # those go in a df, they have to be of the same size, but clearly  the diff will be off by 1
    # sanity logic, and probably use the builint hist for this
    bin_widths = np.concatenate(([0.0], bin_widths))
    hist = np.append(hist, 0.0)  # type: ignore
    return hist, bin_edges, bin_widths  # type: ignore


def count_(
    df: pd.DataFrame,  # type: ignore
    x_col: str,
    name: str,
    weights: Optional["GGScale"] = None,
) -> pd.DataFrame:
    # TODO critical, medium complexity
    # we rename to counts_GGPLOTNIM_INTERNAL
    # need to make a choice here
    if weights is None:
        result = df[x_col].value_counts().reset_index()
        result = result[[x_col, "count"]].rename({"count": name})
    else:
        result = df.groupby(x_col)[weights.get_col_name()].sum().reset_index()  # type: ignore
        result = result[[x_col, "count"]].rename({"count": name})

    return result


def post_process_scales(filled_scales: "FilledScales", plot: "GgPlot"):
    from python_ggplot.gg.ticks import get_x_ticks, get_y_ticks

    """
    TODO refactor this#
    we need something like geom.fill() with mixins
    make it work first
    """
    x_scale: Optional[Scale] = None
    y_scale: Optional[Scale] = None

    for geom in plot.geoms:
        geom_data = geom.gg_data.data
        geom_data = geom_data or plot.data.copy(deep=False)
        df = geom_data
        filled_geom = None

        if geom.geom_type in [
            GeomType.POINT,
            GeomType.LINE,
            GeomType.ERROR_BAR,
            GeomType.TILE,
            GeomType.TEXT,
            GeomType.RASTER,
        ]:
            # can be handled the same
            # need x and y data for sure
            if geom.stat_type == StatType.IDENTITY:
                filled_geom = filled_identity_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.COUNT:
                filled_geom = filled_count_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.SMOOTH:
                filled_geom = filled_smooth_geom(df, geom, filled_scales)
            else:
                filled_geom = filled_bin_geom(df, geom, filled_scales)

        elif geom.geom_type in [GeomType.HISTOGRAM, GeomType.FREQ_POLY]:
            if geom.stat_type == StatType.IDENTITY:
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, geom, filled_scales)
                # still a histogram like geom, make sure bottom is still at 0!
                if filled_geom.gg_data.y_scale is None:
                    raise GGException("expected y_scale")

                filled_geom.gg_data.y_scale = Scale(
                    low=min(0.0, filled_geom.gg_data.y_scale.low),
                    high=filled_geom.gg_data.y_scale.high,
                )
            elif geom.stat_type == StatType.BIN:
                # calculate histogram
                filled_geom = filled_bin_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.COUNT:
                raise GGException(
                    "For discrete counts of your data use " "`geom_bar` instead!"
                )
            elif geom.stat_type == StatType.SMOOTH:
                raise GGException(
                    "Smoothing statistics not implemented for histogram & frequency polygons. "
                    "Do you want a `density` plot using `geom_density` instead?"
                )

        elif geom.geom_type == GeomType.BAR:
            if geom.stat_type == StatType.IDENTITY:
                # essentially take same data as for point
                filled_geom = filled_identity_geom(df, geom, filled_scales)
                # still a geom_bar, make sure bottom is still at 0!
                if filled_geom.gg_data.y_scale is None:
                    raise GGException("expected y scale")

                filled_geom.gg_data.y_scale = Scale(
                    low=min(0.0, filled_geom.gg_data.y_scale.low),
                    high=filled_geom.gg_data.y_scale.high,
                )
            elif geom.stat_type == StatType.COUNT:
                # count values in classes
                filled_geom = filled_count_geom(df, geom, filled_scales)
            elif geom.stat_type == StatType.BIN:
                raise GGException(
                    "For continuous binning of your data use "
                    "`geom_histogram` instead!"
                )
            elif geom.stat_type == StatType.SMOOTH:
                raise GGException(
                    "Smoothing statistics not supported for bar plots. Do you want a "
                    "`density` plot using `geom_density` instead?"
                )

        if filled_geom is None:
            raise GGException("filled geom should not be none")

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

    filled_scales.x_scale = final_x_scale
    filled_scales.y_scale = final_y_scale


def add_counts_by_position(
    col_sum: pd.Series,  # type: ignore
    col: pd.Series,  # type: ignore
    pos: Optional[PositionType],
) -> pd.Series:
    # TODO use is_numeric_dtype in other places of the code base
    if pd.api.types.is_numeric_dtype(col):  # type: ignore
        if pos == PositionType.STACK:
            if len(col_sum) == 0:
                return col.copy()
            else:
                return col_sum + col
        elif pos in (PositionType.IDENTITY, PositionType.DODGE):
            return col.copy()
        elif pos == PositionType.FILL:
            return pd.Series([1.0])
        else:
            raise GGException("unexpected position type")
    else:
        return col.copy()


def split_discrete_set_map(
    df: pd.DataFrame, scales: List["GGScale"]  # type: ignore
) -> Tuple[List[str], List[str]]:
    set_disc_cols: List[str] = []
    map_disc_cols: List[str] = []

    for scale in scales:
        # TODO URGENT easy fix
        # Original implementation checks if its constant
        if str(scale.gg_data.col) in df.columns:
            map_disc_cols.append(str(scale.gg_data.col))
        else:
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


def call_smoother(
    fg: FilledGeom, df: pd.DataFrame, scale: "GGScale", range: Any
) -> NDArray[np.floating[Any]]:

    geom = fg.gg_data.geom
    stat_kind = geom.gg_data.stat_kind
    if not isinstance(stat_kind, StatSmooth):
        raise GGException("stat type has to be smooth to call smooth function")

    data = df[scale.get_col_name()]  # type: ignore

    if stat_kind.method_type == SmoothMethodType.SVG:
        # TODO we need to convert the result to np.array float
        # smoothing is lower priority, so for now we are fine without it
        return stat_kind.svg_smooth(data)  # type: ignore

    elif stat_kind.method_type == SmoothMethodType.POLY:
        x_data = df[fg.gg_data.x_col]  # type: ignore
        return stat_kind.polynomial_smooth(x_data, data)  # type: ignore

    elif stat_kind.method_type == SmoothMethodType.LM:
        raise GGException("Levenberg-Marquardt fitting is not implemented yet.")

    raise GGException("Unknown smoothing method")


def _get_scale_col_name(scale: Optional["GGScale"]) -> Optional[str]:
    if scale is None:
        return None
    return scale.get_col_name()


def _get_filled_geom_from_scale(scale: Optional["GGScale"]):
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


def filled_identity_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: "FilledScales"
) -> FilledGeom:
    from python_ggplot.gg.styles.utils import apply_style

    """
    TODO refactor/test/fix this
    """
    x, y, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)

    x_col = _get_scale_col_name(x)
    y_col = _get_scale_col_name(y)

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x_col,
        y_col=y_col,
        x_scale=determine_data_scale(x, cont, df),
        y_scale=determine_data_scale(y, cont, df),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        x_discrete_kind=_get_filled_geom_from_scale(x),
        y_discrete_kind=_get_filled_geom_from_scale(y),
        num_x=0,
        num_y=0,
    )

    fg = FilledGeom(gg_data=fg_data)
    result, df = create_filled_geom(fg, filled_scales, geom.geom_type, df)

    # TODO this has to change, but is fine for now
    style = GGStyle()

    # Apply style for set values
    apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        grouped = df.groupby(map_disc_cols, sort=True)  # type: ignore
        col = pd.Series(dtype=float)  # type: ignore
        # TODO this needs fixing, ignore types for now, keep roughly working logic
        for keys, sub_df in grouped:  # type: ignore
            if len(keys) > 1:
                raise GGException("we assume this is 1")

            apply_style(style, sub_df, discretes, [(keys[0], VString(i)) for i in grouped.groups])  # type: ignore

            yield_df = sub_df.copy()
            if x is None:
                # we should have not reached this point, but raise here for now
                raise GGException("x scale is None")

            x.set_x_attributes(result, yield_df)

            if geom.gg_data.position == PositionType.STACK:
                yield_df[PREV_VALS_COL] = 0.0 if len(col) == 0 else col.copy()  # type: ignore

            col = add_counts_by_position(
                yield_df[result.gg_data.y_col],  # type: ignore
                col,  # type: ignore
                geom.gg_data.position,
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df[result.gg_data.y_col] = col

            yield_df = result.maybe_filter_unique(yield_df)
            # this has to be copied otherwise the same style is changed
            base_style = deepcopy(style)
            style_, styles_, temp_yield_df = apply_cont_scale_if_any(
                yield_df, cont, base_style, geom.geom_type, to_clone=True
            )
            result.gg_data.yield_data[keys] = (style_, styles_, temp_yield_df)  # type: ignore

        if geom.gg_data.position == PositionType.STACK and result.is_discrete_y():
            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=result.gg_data.y_scale.low, high=col.max())  # type: ignore
            )

        if (
            geom.geom_type == GeomType.HISTOGRAM
            and geom.gg_data.position == PositionType.STACK
            and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.OUTLINE
        ):
            result.gg_data.yield_data = dict(reversed(list(result.gg_data.yield_data.items())))  # type: ignore
    else:
        yield_df = df.copy()
        yield_df[PREV_VALS_COL] = 0.0
        yield_df = result.maybe_filter_unique(yield_df)
        if x is None:
            # we should have not reached this point, but raise here for now
            raise GGException("x scale is None")
        x.set_x_attributes(result, yield_df)
        key = ("", None)
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )

    if y is not None and y.is_discrete():
        # TODO fix
        # y.label_seqwill exist since is discrete, but this needs refactor anyway
        result.gg_data.y_discrete_kind.label_seq = y.gg_data.discrete_kind.label_seq  # type: ignore

    result.gg_data.num_y = result.gg_data.num_x
    return result


def filled_count_geom(df: pd.DataFrame, geom: Any, filled_scales: Any) -> FilledGeom:
    from python_ggplot.gg.styles.utils import apply_style

    """
    todo refactor the whole function and re use the code
    """
    x, _, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )

    if x.is_continuous():
        raise GGException("For continuous data columns use `geom_histogram` instead!")

    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)
    x_col = x.get_col_name()

    if x.is_discrete():
        # TODO critical, easy task
        # double check if we need to pass empty label_seq
        # or if we need x.gg_data.discrete_kind.label_seq
        x_discrete_kind = FilledGeomDiscrete(label_seq=[])
    else:
        x_discrete_kind = FilledGeomContinuous()

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x_col,
        y_col=COUNT_COL,
        x_scale=determine_data_scale(x, cont, df),
        y_scale=encompassing_data_scale(cont, AxisKind.Y),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        x_discrete_kind=x_discrete_kind,
        y_discrete_kind=FilledGeomContinuous(),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    fg = FilledGeom(gg_data=fg_data)
    result, df = create_filled_geom(fg, filled_scales, geom.geom_type, df)

    all_classes = df[x_col].unique()  # type: ignore
    style = GGStyle()

    apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        grouped = df.groupby(map_disc_cols, sort=False)  # type: ignore
        col = pd.Series(dtype=float)  # For stacking

        if len(cont) > 0:
            raise GGException("cont >0")

        for keys, sub_df in grouped:  # type: ignore
            apply_style(style, sub_df, discretes, [(keys[0], VString(i)) for i in grouped.groups])  # type: ignore

            weight_scale = filled_scales.get_weight_scale(geom, optional=True)
            yield_df = count_(sub_df, x_col, "", weight_scale)

            add_zero_keys(yield_df, all_classes, x_col, "count")  # type: ignore
            yield_df = yield_df.sort_values(x_col)  # type: ignore

            if geom.gg_data.position == PositionType.STACK:
                yield_df["prev_vals"] = 0.0 if len(col) == 0 else col.copy()

            col = add_counts_by_position(
                col, yield_df["count"], geom.position  # type: ignore
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df["count"] = col

            yield_df = result.maybe_filter_unique(yield_df)

            result.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.kind, to_clone=True
            )

            if x is None:
                # we should have not reached this point, but raise here for now
                raise GGException("x scale is None")
            x.set_x_attributes(result, yield_df)

            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=0.0, high=float(col.max()))  # type: ignore
            )
    else:
        if len(cont) > 0:
            raise GGException("cont > 0")

        weight_scale = filled_scales.get_weight_scale(geom, optional=True)
        yield_df = count_(df, x_col, COUNT_COL, weight_scale)
        # TODO double check prev_vals
        yield_df[PREV_VALS_COL] = 0.0

        key = ("", VNull())
        yield_df = result.maybe_filter_unique(yield_df)
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )
        if x is None:
            # we should have not reached this point, but raise here for now
            raise GGException("x scale is None")
        x.set_x_attributes(result, yield_df)
        result.gg_data.y_scale = result.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(yield_df[COUNT_COL].max()))  # type: ignore
        )

    result.gg_data.num_y = round(result.gg_data.y_scale.high)
    result.gg_data.num_x = len(all_classes)  # type: ignore

    if result.gg_data.num_x != len(all_classes):  # type: ignore
        # todo provide better messages...
        raise GGException("ERROR")

    return result


def filled_bin_geom(df: pd.DataFrame, geom: Geom, filled_scales: "FilledScales"):
    from python_ggplot.gg.styles.utils import apply_style

    """
    todo refactor the whole function and re use the code
    """

    stat_kind = geom.gg_data.stat_kind
    # TODO double check if this was the intention, but i think it is
    if getattr(stat_kind, "density", False):
        count_col = "density"
    else:
        count_col = COUNT_COL

    width_col = "binWidths"

    x, _, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales, y_is_none=True
    )

    if x.is_discrete():
        raise GGException("For discrete data columns use `geom_bar` instead!")

    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)

    fg_data = FilledGeomData(
        geom=geom,  # we could do a deep copy on this
        x_col=x.get_col_name(),
        y_col=count_col,
        x_scale=encompassing_data_scale(cont, AxisKind.X),
        y_scale=encompassing_data_scale(cont, AxisKind.Y),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        x_discrete_kind=FilledGeomContinuous(),
        y_discrete_kind=FilledGeomContinuous(),
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    result = FilledGeom(gg_data=fg_data)

    fg = FilledGeom(gg_data=fg_data)
    result, df = create_filled_geom(fg, filled_scales, geom.geom_type, df)

    style = GGStyle()
    apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if map_disc_cols:
        grouped = df.groupby(map_disc_cols, sort=True)  # type: ignore TODO
        col = pd.Series(dtype=float)

        # for keys, sub_df in df: df.sort_values(ascending=False)
        for keys, sub_df in grouped:  # type: ignore
            # now consider settings
            apply_style(style, sub_df, discretes, [(keys[0], VString(i)) for i in grouped.groups])  # type: ignore
            # before we assign calculate histogram
            hist, bins, _ = call_histogram(
                geom,
                sub_df,  # type: ignore
                x,
                filled_scales.get_weight_scale(geom, optional=True),
                x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
            )

            yield_df = pd.DataFrame({x.get_col_name(): bins, count_col: hist})

            if geom.gg_data.position == PositionType.STACK:
                yield_df[PREV_VALS_COL] = col if len(col) > 0 else 0.0

            col = add_counts_by_position(col, pd.Series(hist), geom.gg_data.position)

            if geom.gg_data.position == PositionType.STACK:
                if not (
                    (
                        geom.geom_type == GeomType.HISTOGRAM
                        and geom.gg_data.histogram_drawing_style
                        == HistogramDrawingStyle.BARS
                    )
                    or (geom.geom_type == GeomType.BAR)
                ):
                    yield_df[result.gg_data.y_col] = col

            yield_df = result.maybe_filter_unique(yield_df)
            result.gg_data.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.geom_type, to_clone=True  # type: ignore
            )

            result.gg_data.num_x = max(result.gg_data.num_x, len(yield_df))

            if geom.geom_type == GeomType.FREQ_POLY:
                bin_width = float(bins[1] - bins[0]) if len(bins) > 1 else 0.0
                result.gg_data.x_scale = result.gg_data.x_scale.merge(
                    Scale(
                        low=float(min(bins)) - bin_width / 2.0,
                        high=float(max(bins)) + bin_width / 2.0,
                    )
                )
            else:
                result.gg_data.x_scale = result.gg_data.x_scale.merge(
                    Scale(low=float(min(bins)), high=float(max(bins)))
                )

            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=0.0, high=float(col.max()))  # type: ignore
            )
    else:
        hist, bins, bin_widths = call_histogram(
            geom,
            df,
            x,
            filled_scales.get_weight_scale(geom, optional=True),
            x.gg_data.discrete_kind.data_scale,  # type: ignore TODO
        )

        yield_df = pd.DataFrame(
            {x.get_col_name(): bins, count_col: hist, width_col: bin_widths}
        )
        yield_df[PREV_VALS_COL] = 0.0
        yield_df = result.maybe_filter_unique(yield_df)

        key = ("", VNull())

        if len(cont) != 0:
            raise GGException("seems the data is discrete")

        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )
        result.gg_data.num_x = len(yield_df)
        result.gg_data.x_scale = result.gg_data.x_scale.merge(
            Scale(low=float(min(bins)), high=float(max(bins)))
        )

        result.gg_data.y_scale = result.gg_data.y_scale.merge(
            Scale(low=0.0, high=float(max(hist)))
        )

    result.gg_data.num_y = round(result.gg_data.y_scale.high)

    if x.is_discrete():
        # TODO fix, this is an error
        result.gg_data.x_label_seq = x.gg_data.label_seq  # type: ignore

    return result


def filled_smooth_geom(
    df: pd.DataFrame, geom: Geom, filled_scales: "FilledScales"
) -> FilledGeom:
    from python_ggplot.gg.styles.utils import apply_style

    """
    TODO complete refactor
    reuse logic with filled_identity_geom
    doesnt make a difference for now,
    need a draft version of this to get all the unit tests running
    """

    x, y, discretes, cont = separate_scales_apply_transofrmations(
        df, geom, filled_scales
    )
    set_disc_cols, map_disc_cols = split_discrete_set_map(df, discretes)

    if x.is_discrete():
        raise GGException("expected continuous data")

    if y is not None and y.is_discrete():
        raise GGException("expected continuous data")

    if y is None:
        # TODO i think this logic is wrong, double check
        raise GGException("y is none")

    fg_data = FilledGeomData(
        geom=geom,
        x_col=x.get_col_name(),
        y_col=SMOOTH_VALS_COL,
        x_scale=determine_data_scale(x, cont, df),
        y_scale=determine_data_scale(y, cont, df),
        x_discrete_kind=FilledGeomContinuous(),
        y_discrete_kind=FilledGeomContinuous(),
        # not explicitly passed at initialisisation, we set some defaults
        # TODO investiage if needed
        reversed_x=False,
        reversed_y=False,
        yield_data={},  # type: ignore
        num_x=0,
        num_y=0,
    )
    fg = FilledGeom(gg_data=fg_data)
    result, df = create_filled_geom(fg, filled_scales, geom.geom_type, df)

    style = GGStyle()
    apply_style(style, df, discretes, [(col, VNull()) for col in set_disc_cols])

    if len(map_disc_cols) > 0:
        grouped = df.groupby(map_disc_cols, sort=True)  # type: ignore
        col = pd.Series(dtype=float)  # type: ignore

        for keys, sub_df in grouped:  # type: ignore
            apply_style(style, sub_df, discretes, [(keys[0], VString(i)) for i in grouped.groups])  # type: ignore
            yield_df = sub_df.copy()  # type: ignore

            smoothed = call_smoother(
                result,
                yield_df,  # type: ignore
                y,
                # This has to be continuous for data scale to exist needs cleanup
                range=x.gg_data.discrete_kind.data_scale,  # type: ignore
            )
            yield_df[SMOOTH_VALS_COL] = smoothed
            if x is None:
                # we should have not reached this point, but raise here for now
                raise GGException("x scale is None")
            x.set_x_attributes(result, yield_df)

            if geom.gg_data.position == PositionType.STACK:
                yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index) if len(col) == 0 else col.copy()  # type: ignore

            # possibly modify `col` if stacking
            # TODO double check this
            yield_df[result.gg_data.y_col] = add_counts_by_position(
                yield_df[result.gg_data.y_col],  # type: ignore
                col,  # type: ignore
                geom.gg_data.position,
            )

            if geom.gg_data.position == PositionType.STACK and not (
                (
                    geom.geom_type == GeomType.HISTOGRAM
                    and geom.gg_data.histogram_drawing_style
                    == HistogramDrawingStyle.BARS
                )
                or (geom.geom_type == GeomType.BAR)
            ):
                yield_df[result.y_col] = col  # type: ignore

            yield_df = result.maybe_filter_unique(yield_df)
            result.yield_data[keys] = apply_cont_scale_if_any(  # type: ignore
                yield_df, cont, style, geom.geom_type, to_clone=True  # type: ignore
            )

        if geom.gg_data.position == PositionType.STACK and not result.is_discrete_y():
            # only update required if stacking, as we've computed the range beforehand
            result.gg_data.y_scale = result.gg_data.y_scale.merge(
                Scale(low=result.gg_data.y_scale.low, high=col.max())  # type: ignore
            )

        if (
            geom.geom_type == GeomType.HISTOGRAM
            and geom.gg_data.position == PositionType.STACK
            and geom.gg_data.histogram_drawing_style == HistogramDrawingStyle.OUTLINE
        ):
            result.gg_data.yield_data = dict(reversed(list(result.gg_data.yield_data.items())))  # type: ignore
    else:
        yield_df = df.copy()
        smoothed = call_smoother(
            result, yield_df, y, range=x.data_scale  # type: ignore TODO critical FIX
        )
        yield_df[PREV_VALS_COL] = pd.Series(0.0, index=yield_df.index)  # type: ignore
        yield_df[SMOOTH_VALS_COL] = smoothed
        yield_df = result.maybe_filter_unique(yield_df)

        if x is None:
            # we should have not reached this point, but raise here for now
            raise GGException("x scale is None")
        x.set_x_attributes(result, yield_df)

        key = ("", VNull())
        result.gg_data.yield_data[key] = apply_cont_scale_if_any(  # type: ignore
            yield_df, cont, style, geom.geom_type
        )

    result.gg_data.num_y = result.gg_data.num_x

    return result
