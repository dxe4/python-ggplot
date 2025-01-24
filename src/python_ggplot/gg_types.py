from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from python_ggplot.core.objects import (
    AxisKind,
    Color,
    ErrorBarKind,
    Font,
    LineType,
    MarkerKind,
    Scale,
    TexOptions,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.datamancer_pandas_compat import FormulaNode, GGValue

COUNT_COL = "counts_GGPLOTNIM_INTERNAL"
PREV_VALS_COL = "prevVals_GGPLOTNIM_INTERNAL"
SMOOTH_VALS_COL = "smoothVals_GGPLOTNIM_INTERNAL"

if TYPE_CHECKING:
    from python_ggplot.gg_geom import FilledScales
    from python_ggplot.gg_scales import ColorScale, ScaleFreeKind, ScaleKind, ScaleValue

    # TODO view port we should be able to import, this shouldnt be here, but adding temporarily
    from python_ggplot.graphics.views import ViewPort


class AestheticError(Exception):
    pass


class PositionType(Enum):
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
class StatKind(ABC):

    @property
    @abstractmethod
    def stat_type(self) -> StatType:
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
DiscreteFormat = Callable[[GGValue], str]
ContinuousFormat = Callable[[float], str]


@dataclass
class Aesthetics:
    scale_kind: "ScaleKind"
    position_kind: PositionType
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


class BinPositionType(Enum):
    NONE = auto()
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()


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


@dataclass
class Ridges:
    col: FormulaNode
    overlap: float
    show_ticks: bool
    label_order: Dict[GGValue, int]


@dataclass
class Draw:
    fname: str
    width: Optional[float] = None
    height: Optional[float] = None
    # LATER: wont use TEX for now
    tex_options: Optional[TexOptions] = None
    backend: Optional[str] = None


@dataclass
class GGStyle:
    color: Optional[Color] = None
    size: Optional[float] = None
    line_type: Optional[LineType] = None
    line_width: Optional[float] = None
    fill_color: Optional[Color] = None
    marker: Optional[MarkerKind] = None
    error_bar_kind: Optional[ErrorBarKind] = None
    alpha: Optional[float] = None
    font: Optional[Font] = None


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
class Facet:
    columns: List[str]
    scale_free_kind: "ScaleFreeKind"


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
    label: "GGValue"


@dataclass
class ThemeMarginLayout:
    left: Quantity
    right: Quantity
    top: Quantity
    bottom: Quantity
    requires_legend: bool


@dataclass
class JsonDummyDraw:
    """
    LATER: wont implement for now
    """

    fname: str
    width: Optional[float]
    height: Optional[float]
    backend: str  # we only support cairo for now


MainAddScales = Tuple[Optional[Scale], List[Scale]]


@dataclass
class PlotView:
    filled_scales: "FilledScales"
    view: "ViewPort"


class VegaError(Exception):
    pass


class VegaBackend(str, Enum):
    WEBVIEW = auto()
    BROWSER = auto()


@dataclass
class VegaDraw:
    """
    # LATER: no need to start with vega, finish cairo first
    """

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
class VegaTex:
    """
    LATER: wont implement for now
    """

    fname: str
    width: Optional[float]
    height: Optional[float]
    tex_options: TexOptions
