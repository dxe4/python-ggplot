from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from python_ggplot.common.maths import poly_fit, savitzky_golay
from python_ggplot.core.objects import (
    AxisKind,
    Color,
    ErrorBarKind,
    Font,
    GGEnum,
    GGException,
    LineType,
    MarkerKind,
    Scale,
    TexOptions,
    TextAlignKind,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.gg.datamancer_pandas_compat import GGValue, VectorCol, VNull
from python_ggplot.gg.geom import Geom
from python_ggplot.gg.scales.base import (
    GGScale,
    GGScaleData,
    GGScaleDiscrete,
    LinearAndTransformScaleData,
    LinearDataScale,
)

COUNT_COL = "counts_GGPLOTNIM_INTERNAL"
PREV_VALS_COL = "prevVals_GGPLOTNIM_INTERNAL"
SMOOTH_VALS_COL = "smoothVals_GGPLOTNIM_INTERNAL"

if TYPE_CHECKING:
    from python_ggplot.gg.scales import FilledScales, ScaleFreeKind

    # TODO view port we should be able to import, this shouldnt be here, but adding temporarily
    from python_ggplot.graphics.views import ViewPort


class AestheticError(Exception):
    pass


class PositionType(GGEnum):
    IDENTITY = auto()
    STACK = auto()
    DODGE = auto()
    FILL = auto()


class StatType(GGEnum):
    IDENTITY = auto()
    COUNT = auto()
    BIN = auto()
    SMOOTH = auto()


@dataclass(frozen=True)
class StatKind(ABC):

    @property
    @abstractmethod
    def stat_type(self) -> StatType:
        pass

    @staticmethod
    def create_from_enum(
        stat_type: StatType, data: Optional[Dict[Any, Any]] = None
    ) -> "StatKind":
        classes = {
            StatType.BIN: StatBin,
            StatType.IDENTITY: StatIdentity,
            StatType.COUNT: StatCount,
            StatType.SMOOTH: StatSmooth,
        }
        if data is None:
            data = {}
        return classes[stat_type](**data)


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
    bin_by: "BinByType"
    density: bool

    @property
    def stat_type(self) -> StatType:
        return StatType.BIN


@dataclass(frozen=True)
class StatSmooth(StatKind):
    span: float
    poly_order: int
    method_type: "SmoothMethodType"

    def polynomial_smooth(self, x: pd.Series[float], y: pd.Series[float]):
        return poly_fit(
            x.to_numpy(),  # type: ignore
            y.to_numpy(),  # type: ignore
            self.poly_order,
        )

    def svg_smooth(self, data: pd.Series[float]):
        window_size = round(len(data) * self.span)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 1 or window_size > len(data):
            raise GGException(
                f"The given `span` value results in a "
                f"Savitzky-Golay filter window of {window_size} for input "
                f"data with length {len(data)}."
            )
        # TODO high priority, this can return int or poly1d
        return savitzky_golay(
            data.to_numpy(), window_size, self.poly_order  # type: ignore
        )

    @property
    def stat_type(self) -> StatType:
        return StatType.SMOOTH


class DiscreteType(GGEnum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class DiscreteKind:
    pass


# todo refactor
DiscreteFormat = Callable[[GGValue], str]
ContinuousFormat = Callable[[float], str]


@dataclass
class Aesthetics:
    x: Optional["GGScale"] = None
    x_min: Optional["GGScale"] = None
    x_max: Optional["GGScale"] = None
    y: Optional["GGScale"] = None
    y_min: Optional["GGScale"] = None
    y_max: Optional["GGScale"] = None
    fill: Optional["GGScale"] = None
    color: Optional["GGScale"] = None
    alpha: Optional["GGScale"] = None
    size: Optional["GGScale"] = None
    shape: Optional["GGScale"] = None
    width: Optional["GGScale"] = None
    height: Optional["GGScale"] = None
    text: Optional["GGScale"] = None
    y_ridges: Optional["GGScale"] = None
    weight: Optional["GGScale"] = None


@dataclass
class SecondaryAxis:
    name: str
    scale: GGScale
    # TODO i dont like this, but thats how its inherited
    # id rather be explicit here makes it more understandable
    axis_kind: AxisKind = AxisKind.X


discrete_format = Callable[[Union[int, str]], str]
continuous_format = Callable[[float], str]


class DateTickAlgorithmType(GGEnum):
    FILTER = auto()
    ADD_DURATION = auto()
    CUSTOM_BREAKS = auto()


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


class DataType(GGEnum):
    MAPPING = auto()
    SETTING = auto()
    # TODO high priority this shouldnt be there, but is because nim
    #  doesnt pass it in update_aes_ridges
    NULL = auto()


class BinPositionType(GGEnum):
    NONE = auto()
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()


class SmoothMethodType(str, GGEnum):
    SVG = auto()
    LM = auto()
    POLY = auto()


class BinByType(str, GGEnum):
    FULL = auto()
    SUBSET = auto()


class OutsideRangeKind(str, GGEnum):
    NONE = auto()
    DROP = auto()
    CLIP = auto()


@dataclass
class Ridges:
    col: VectorCol
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
    x_margin_range: Scale = Scale(low=0.0, high=0.0)  # TODO double check
    y_margin_range: Scale = Scale(low=0.0, high=0.0)  # TODO double check
    x_ticks_rotate: float = 0.0
    y_ticks_rotate: float = 0.0
    x_ticks_text_align: TextAlignKind = TextAlignKind.LEFT
    y_ticks_text_align: TextAlignKind = TextAlignKind.LEFT

    base_font_size: Optional[float] = None
    sub_title_font: Optional[Dict[Any, Any]] = None
    tick_label_font: Optional[Font] = None
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

    x_tick_label_margin: Optional[float] = None

    y_tick_label_margin: Optional[float] = None
    legend_position: Optional[Tuple[float]] = None
    legend_order: Optional[List[int]] = None
    hide_legend: Optional[bool] = None
    canvas_color: Optional[Color] = None
    plot_background_color: Optional[Color] = None
    grid_lines: Optional[bool] = None
    grid_line_color: Optional[Color] = None
    grid_line_width: Optional[float] = None
    minor_grid_lines: Optional[bool] = None
    minor_grid_line_width: Optional[float] = None
    only_axes: Optional[bool] = None
    discrete_scale_margin: Optional[Quantity] = None
    x_range: Optional[Scale] = None
    y_range: Optional[Scale] = None
    x_margin: Optional[float] = None
    y_margin: Optional[float] = None
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
    aes: Aesthetics
    theme: Theme
    backend: str = field(default="cairo")  # Will be cairo only for a while..
    title: Optional[str] = None
    sub_title: Optional[str] = None
    facet: Optional[Any] = None
    ridges: Optional[Ridges] = None
    geoms: Optional[List[Geom]] = None
    annotations: Optional[List[Any]] = None

    def update_aes_ridges(self: "GgPlot") -> "GgPlot":
        """
        TODO high priority this seems wrong on nim side
        how come we have:
            a) dcKind: dcDiscrete
            b) no valueMap, labelSeq, formatDiscreteLabel
        this should not be possible
        we will pass {} and [] for now,
        dont want to set them as optioanl it has a cascading effect on doing null checks

        Scale(
            scKind: scLinearData,
            col: ridge.col,
            axKind: akY,
            hasDiscreteness: true, # force scale to be discrete!
            dcKind: dcDiscrete,
            ids: {0'u16 .. high(uint16)}
        )
        case dcKind*: DiscreteKind
        of dcDiscrete:
          valueMap*: OrderedTable[Value, ScaleValue]
          labelSeq*: seq[Value]
          formatDiscreteLabel*: DiscreteFormat
        """
        if self.ridges is None:
            raise GGException("expected ridges")

        ridge = self.ridges
        data = LinearAndTransformScaleData(
            # TODO, reversed and transform are required,
            # but update ridges doesn't explicitly set them
            # why ?
            axis_kind=AxisKind.Y,
            reversed=False,
            transform=lambda x: x,
        )
        gg_data = GGScaleData(
            col=ridge.col,
            has_discreteness=True,
            discrete_kind=GGScaleDiscrete(value_map={}, label_seq=[]),  # type: ignore
            ids=set(range(65536)),
            data_type=DataType.NULL,
            value_kind=VNull(),
        )
        scale = LinearDataScale(
            gg_data=gg_data,
            data=data,
        )

        self.aes.y_ridges = scale
        return self


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


class VegaBackend(str, GGEnum):
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
