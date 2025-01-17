from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import pandas as pd

from python_ggplot.core.objects import (
    AxisKind,
    Color,
    ErrorBarKind,
    Font,
    GGException,
    LineType,
    MarkerKind,
    Scale,
    TexOptions,
)
from python_ggplot.core.units.objects import Quantity

COUNT_COL = "counts_GGPLOTNIM_INTERNAL"
PREV_VALS_COL = "prevVals_GGPLOTNIM_INTERNAL"
SMOOTH_VALS_COL = "smoothVals_GGPLOTNIM_INTERNAL"

if TYPE_CHECKING:
    from python_ggplot.gg_scales import ColorScale, ScaleFreeKind, ScaleKind, ScaleValue

    # TODO view port we should be able to import, this shouldnt be here, but adding temporarily
    from python_ggplot.graphics.view import ViewPort


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


class FormulaNode:
    pass


class AestheticError(Exception):
    pass


class PositionKind(Enum):
    IDENTITY = auto()
    STACK = auto()
    DODGE = auto()
    FILL = auto()


class StatType(Enum):
    IDENTITY = auto()
    COUNT = auto()
    BIN = auto()
    SMOOTH = auto()


@dataclass
class StatKind:
    pass


class StatIdentity(StatKind):
    @property
    def stat_type(self) -> StatType:
        return StatType.IDENTITY


class StatCount(StatKind):
    @property
    def stat_type(self) -> StatType:
        return StatType.COUNT


class StatBin(StatKind):
    num_bins: int
    bin_width: Optional[float] = None
    bin_edges: Optional[List[float]] = None
    bin_by: "BinByKind"
    density: bool

    @property
    def stat_type(self) -> StatType:
        return StatType.BIN


class StatSmooth(StatKind):
    span: float
    poly_oder: int
    method_kind: "SmoothMethodKind"

    @property
    def stat_type(self) -> StatType:
        return StatType.SMOOTH


class DiscreteType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class DiscreteKind:
    pass


# todo refactor
DiscreteFormat = Callable[["Value"], str]
ContinuousFormat = Callable[[float], str]


@dataclass
class Aesthetics:
    scale_kind: "ScaleKind"
    position_kind: PositionKind
    stat_kind: StatKind
    discrete_kind: DiscreteKind
    x: Optional["ScaleKind"] = None
    x_min: Optional["ScaleKind"] = None
    x_max: Optional["ScaleKind"] = None
    y: Optional["ScaleKind"] = None
    y_min: Optional["ScaleKind"] = None
    y_max: Optional["ScaleKind"] = None
    fill: Optional["ScaleKind"] = None
    color: Optional["ScaleKind"] = None
    alpha: Optional["ScaleKind"] = None
    size: Optional["ScaleKind"] = None
    shape: Optional["ScaleKind"] = None
    width: Optional["ScaleKind"] = None
    height: Optional["ScaleKind"] = None
    text: Optional["ScaleKind"] = None
    y_ridges: Optional["ScaleKind"] = None
    weight: Optional["ScaleKind"] = None


@dataclass
class SecondaryAxis:
    name: str
    axis_kind: AxisKind
    scale_kind: "ScaleKind"


discrete_format = Callable[[Union[int, str]], str]
continuous_format = Callable[[float], str]


class DateTickAlgorithmKind(Enum):
    # Compute date ticks by filtering to closest matches
    DTA_FILTER = auto()
    # Compute date ticks by adding given duration to start time
    DTA_ADD_DURATION = auto()
    # Use user-given custom breaks (as UNIX timestamps)
    DTA_CUSTOM_BREAKS = auto()


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
class DataKind:
    mapping: str = "mapping"
    setting: str = "setting"


class BinPositionKind(str, Enum):
    NONE = auto()
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()


class GeoType:
    GEOM_POINT = auto()
    GEOM_BAR = auto()
    GEOM_HISTOGRAM = auto()
    GEOM_FREQ_POLY = auto()
    GEOM_TILE = auto()
    GEOM_LINE = auto()
    GEOM_ERROR_BAR = auto()
    GEOM_TEXT = auto()
    GEOM_RASTER = auto()


@dataclass
class GeomKind:
    @property
    def geom_type(self):
        raise GGException("not implemented for interface")


class GeomPoint(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_POINT


class GeomBar(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_BAR


class GeomHistogram(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_HISTOGRAM


class GeomFreqPoly(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_FREQ_POLY


class GeomTile(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_TILE


class GeomLine(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_LINE


class GeomErrorBar(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_ERROR_BAR


class GeomText(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_TEXT


class GeomRaster(GeomKind):
    @property
    def geom_type(self):
        return GeoType.GEOM_RASTER


class HistogramDrawingStyle(str, Enum):
    BARS = auto()
    OUTLINE = auto()


class SmoothMethodKind(str, Enum):
    SVG = auto()
    LM = auto()
    POLY = auto()


class BinByKind(str, Enum):
    FULL = auto()
    SUBSET = auto()


class OutsideRangeKind(str, Enum):
    NONE = auto()
    DROP = auto()
    CLIP = auto()


class VegaBackend(str, Enum):
    WEBVIEW = auto()
    BROWSER = auto()


@dataclass
class Ridges:
    col: FormulaNode
    overlap: float
    show_ticks: bool
    label_order: Dict[Value, int]


@dataclass
class Draw:
    fname: str
    width: Optional[float] = None
    height: Optional[float] = None
    tex_options: Optional[TexOptions] = None
    backend: Optional[str] = None


@dataclass
class VegaDraw:
    fname: str
    width: Optional[float] = None
    height: Optional[float] = None
    as_pretty_json: Optional[bool] = None
    show: bool = True
    backend: VegaBackend = VegaBackend.WEBVIEW
    remove_file: bool = False
    div_name: str = "vega-div"
    vega_libs_path: str = "https://cdn.jsdelivr.net/npm/"
    vega_version: str = "5"
    vega_lite_version: str = "4"
    vega_embed_version: str = "6"


@dataclass
class GGStyle:
    color: Optional[str] = None
    size: Optional[float] = None
    line_type: Optional[str] = None
    line_width: Optional[float] = None
    fill_color: Optional[str] = None
    marker: Optional[str] = None
    error_bar_kind: Optional[str] = None
    alpha: Optional[float] = None
    font: Optional[dict] = None


@dataclass
class Theme:
    base_font_size: Optional[float] = None
    sub_title_font: Optional[dict] = None
    tick_label_font: Optional[dict] = None
    hide_ticks: Optional[bool] = None
    hide_tick_labels: Optional[bool] = None
    hide_labels: Optional[bool] = None
    title: Optional[str] = None
    sub_title: Optional[str] = None
    x_label: Optional[str] = None
    x_label_margin: Optional[float] = None
    x_label_secondary: Optional[str] = None
    y_label: Optional[str] = None
    y_label_margin: Optional[float] = None
    y_label_secondary: Optional[str] = None
    x_ticks_rotate: Optional[float] = None
    x_ticks_text_align: Optional[str] = None
    x_tick_label_margin: Optional[float] = None
    y_ticks_rotate: Optional[float] = None
    y_ticks_text_align: Optional[str] = None
    y_tick_label_margin: Optional[float] = None
    legend_position: Optional[tuple] = None
    legend_order: Optional[List[int]] = None
    hide_legend: Optional[bool] = None
    canvas_color: Optional[str] = None
    plot_background_color: Optional[str] = None
    grid_lines: Optional[bool] = None
    grid_line_color: Optional[str] = None
    grid_line_width: Optional[float] = None
    minor_grid_lines: Optional[bool] = None
    minor_grid_line_width: Optional[float] = None
    only_axes: Optional[bool] = None
    discrete_scale_margin: Optional[float] = None
    x_range: Optional[Scale] = None
    y_range: Optional[Scale] = None
    x_margin: Optional[float] = None
    x_margin_range: Optional[Scale] = None
    y_margin: Optional[float] = None
    y_margin_range: Optional[Scale] = None
    x_outside_range: Optional[OutsideRangeKind] = None
    y_outside_range: Optional[OutsideRangeKind] = None
    plot_margin_left: Optional[float] = None
    plot_margin_right: Optional[float] = None
    plot_margin_top: Optional[float] = None
    plot_margin_bottom: Optional[float] = None
    facet_margin: Optional[float] = None
    prefer_rows_over_columns: Optional[bool] = None


@dataclass
class Facet(Enum):
    columns = List[str]
    scale_free_kind = "ScaleFreeKind"


@dataclass
class GgPlot:
    data: pd.DataFrame
    title: str
    sub_title: str
    aes: Aesthetics
    facet: Optional[Any]
    ridges: Optional[Ridges]
    geoms: List[Any]
    annotations: List[Any]
    theme: Theme
    backend: str


@dataclass
class Annotation:
    left: Optional[float]
    bottom: Optional[float]
    x: Optional[float]
    y: Optional[float]
    text: str
    font: "Font"
    rotate: Optional[float]
    background_color: "Color"


@dataclass
class StyleLabel:
    style: "GGStyle"
    label: "Value"


@dataclass
class ThemeMarginLayout:
    left: Quantity
    right: Quantity
    top: Quantity
    bottom: Quantity
    requires_legend: bool


@dataclass
class JsonDummyDraw:
    fname: str
    width: Optional[float]
    height: Optional[float]
    backend: str  # we only support cairo for now


@dataclass
class VegaTex:
    fname: str
    width: Optional[float]
    height: Optional[float]
    tex_options: TexOptions


@dataclass
class Geom:
    gid: int
    kind: GeomKind
    data: Optional[pd.DataFrame] = None
    user_style: Optional["GGStyle"] = None
    position: Optional["PositionKind"] = None
    aes: Optional["Aesthetics"] = None
    bin_position: Optional["BinPositionKind"] = None
    # used for geom_type histogram
    histogram_drawing_style: Optional["HistogramDrawingStyle"] = None

    def __post_init__(self):
        if (
            self.kind.geom_type == GeoType.GEOM_HISTOGRAM
            and not self.histogram_drawing_style
        ):
            raise GGException("histogram geom needs to specify histogram_drawing_style")


class FilledGeomDiscreteKind(DiscreteKind):
    pass


class GGScaleDiscrete(FilledGeomDiscreteKind):
    label_seq: List[Value]

    @property
    def discrete_type(self):
        return DiscreteType.DISCRETE


class GGScaleContinuous(FilledGeomDiscreteKind):
    @property
    def discrete_type(self):
        return DiscreteType.CONTINUOUS


class FilledGeomErrorBar(GeomErrorBar):
    xmin: Optional[str]
    ymin: Optional[str]
    xmax: Optional[str]
    ymax: Optional[str]


@dataclass
class TitleRasterData:
    fill_col: str
    fill_data_scale: Scale
    width: Optional[str]
    height: Optional[str]
    color_scale: "ColorScale"


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
    yield_data: OrderedDict[Value, Tuple[GGStyle, List[GGStyle], pd.DataFrame]]
    num_x: int
    num_y: int
    x_discrete_kind: FilledGeomDiscreteKind
    y_discrete_kind: FilledGeomDiscreteKind


MainAddScales = Tuple[Optional[Scale], List[Scale]]


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


@dataclass
class PlotView:
    filled_scales: FilledScales
    view: "ViewPort"


class VegaError(Exception):
    pass
