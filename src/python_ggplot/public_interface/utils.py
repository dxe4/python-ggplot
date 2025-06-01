"""
TODO this whole file needs cleaning up
"""

import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import numpy as np
import pandas as pd
from numpy._core.numeric import empty

from python_ggplot.common.enum_literals import (
    OUTSIDE_RANGE_KIND_VALUES,
    UNIT_TYPE_VALUES,
)
from python_ggplot.core.chroma import int_to_color, to_opt_color
from python_ggplot.core.coord.objects import (
    CentimeterCoordType,
    Coord,
    Coord1D,
    CoordsInput,
    DataCoord,
    DataCoordType,
    LengthCoord,
    PointCoordType,
    RelativeCoordType,
    StrHeightCoordType,
    StrWidthCoordType,
    TextCoordData,
)
from python_ggplot.core.embed import view_embed_at
from python_ggplot.core.objects import (
    GREY92,
    TRANSPARENT,
    WHITE,
    AxisKind,
    Color,
    Font,
    GGException,
    Gradient,
    LineType,
    Scale,
    Style,
    TextAlignKind,
    UnitType,
)
from python_ggplot.core.units.objects import DataUnit, PointUnit, Quantity, RelativeUnit
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    GGValue,
    VectorCol,
    VFillColor,
    VLinearData,
)
from python_ggplot.gg.drawing import create_gobj_from_geom
from python_ggplot.gg.geom.base import FilledGeom, Geom, GeomType, post_process_scales
from python_ggplot.gg.scales import FillColorScaleValue, ScaleValue
from python_ggplot.gg.scales.base import (
    ColorScale,
    ColorScaleKind,
    FillColorScale,
    FilledScales,
    GGScale,
    GGScaleContinuous,
    GGScaleData,
    GGScaleDiscrete,
    LinearAndTransformScaleData,
    LinearDataScale,
    ScaleFreeKind,
    ScaleType,
    ShapeScale,
    SizeScale,
    TransformedDataScale,
)
from python_ggplot.gg.scales.collect_and_fill import collect_scales
from python_ggplot.gg.theme import (
    build_theme,
    calculate_margin_range,
    get_canvas_background,
    get_grid_line_style,
    get_minor_grid_line_style,
    get_plot_background,
    has_secondary,
)
from python_ggplot.gg.ticks import handle_discrete_ticks, handle_ticks
from python_ggplot.gg.types import (
    Aesthetics,
    Annotation,
    DiscreteFormat,
    DiscreteType,
    Facet,
    GgPlot,
    OutsideRangeKind,
    PlotView,
    PossibleColor,
    Ridges,
    SecondaryAxis,
    Theme,
    ThemeMarginLayout,
)
from python_ggplot.gg.utils import calc_rows_columns, to_opt_sec_axis
from python_ggplot.graphics.draw import background, draw_line, draw_to_file, layout
from python_ggplot.graphics.initialize import (
    InitMultiLineInput,
    InitRectInput,
    InitTextInput,
    TickLabelsInput,
    init_coord_1d_from_view,
    init_grid_lines,
    init_multi_line_text,
    init_point_from_coord,
    init_rect,
    init_text,
    init_ticks,
    tick_labels,
    xlabel,
    ylabel,
)
from python_ggplot.graphics.objects import (
    GOLabel,
    GOPoint,
    GOText,
    GOTickLabel,
    GOType,
    GraphicsObject,
)
from python_ggplot.graphics.views import ViewPort, ViewPortInput

BASE_TO_LOG = {
    10: math.log10,
    2: math.log2,
}


@dataclass
class _LogTrans:
    base: int

    def __call__(self, val: Any) -> Any:
        if math.isclose(val, 0.0):
            return 0.0
        return BASE_TO_LOG[self.base](val)


@dataclass
class _LogInverseTrans:
    base: int

    def __call__(self, val: Any) -> Any:
        if math.isclose(val, 0.0):
            return 0.0
        return math.pow(self.base, val)


def scale_axis_log(
    axis_kind: AxisKind, base: int, breaks: Optional[Union[int, List[float]]] = None
) -> GGScale:

    # TODO this leaves room for errors
    gg_data = GGScaleData(
        col=VectorCol(""),  # will be filled when added to GgPlot obj
        value_kind=VTODO(),  # i guess here same with col, will be added later
        discrete_kind=GGScaleContinuous(),
    )
    scale = TransformedDataScale(
        gg_data=gg_data,
        data=LinearAndTransformScaleData(
            axis_kind=axis_kind,
            transform=_LogTrans(10),
        ),
        transform=_LogTrans(10),
        inverse_transform=_LogInverseTrans(10),
    )
    scale.assign_breaks(breaks or [])
    return scale


def scale_axis_discrete_with_label_fn(
    axis_kind: AxisKind,
    name: str = "",
    labels_fn: Optional[DiscreteFormat] = None,
    sec_axis: Optional[SecondaryAxis] = None,
    reversed: bool = False,
) -> GGScale:
    secondary_axis = to_opt_sec_axis(sec_axis, axis_kind)
    linear_data = LinearAndTransformScaleData(
        axis_kind=axis_kind,
        secondary_axis=secondary_axis,
        reversed=reversed,
    )
    gg_data = GGScaleData(
        col=VectorCol(name),
        value_kind=VTODO(),  # seems it doesnt need one?
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(format_discrete_label=labels_fn),
    )
    scale = LinearDataScale(
        gg_data=gg_data,
        data=linear_data,
    )
    return scale


def scale_axis_discrete_with_labels(
    axis_kind: AxisKind,
    name: str = "",
    labels: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
    reversed: bool = False,
) -> GGScale:

    if labels is None:
        labels = OrderedDict()

    def format_discrete_label_(value: GGValue):
        """
        TODO double check this logic
        original version passes a dict i instead of a function
        i assume dict is callable in nim
        """
        return str(labels[value])

    value_map_ = {VLinearData(data=k): v for k, v in labels.items()}
    label_seq_ = [i for i in labels]

    secondary_axis = to_opt_sec_axis(sec_axis, axis_kind)
    linear_data = LinearAndTransformScaleData(
        axis_kind=axis_kind,
        secondary_axis=secondary_axis,
        reversed=reversed,
    )
    gg_data = GGScaleData(
        col=VectorCol(name),
        value_kind=VTODO(),  # seems it doesnt need one?
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(
            value_map=value_map_,  # type: ignore  TODO check this
            label_seq=label_seq_,
            format_discrete_label=format_discrete_label_,
        ),
    )
    scale = LinearDataScale(
        gg_data=gg_data,
        data=linear_data,
    )
    return scale


def scale_reverse(
    axis_kind: AxisKind,
    name: str = "",
    sec_axis: Optional[SecondaryAxis] = None,
    discrete_kind: DiscreteType = DiscreteType.CONTINUOUS,
) -> GGScale:
    if discrete_kind == DiscreteType.CONTINUOUS:
        discrete_kind_ = GGScaleContinuous()
    elif discrete_kind == DiscreteType.DISCRETE:
        discrete_kind_ = GGScaleDiscrete()
    else:
        raise GGException("unexpected discrete type")

    secondary_axis = to_opt_sec_axis(sec_axis, axis_kind)
    linear_data = LinearAndTransformScaleData(
        axis_kind=axis_kind,
        secondary_axis=secondary_axis,
        reversed=True,
    )
    gg_data = GGScaleData(
        col=VectorCol(name),
        value_kind=VTODO(),  # seems it doesnt need one?
        has_discreteness=True,
        discrete_kind=discrete_kind_,
    )
    scale = LinearDataScale(
        gg_data=gg_data,
        data=linear_data,
    )
    return scale


def scale_color_or_fill_manual(
    cls: Type[FillColorScale] | Type[ColorScaleKind],
    values: OrderedDict[GGValue, Color],
) -> GGScale:
    label_seq = [i for i in values]
    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict(
        (VFillColor(data=k), FillColorScaleValue(color=v)) for k, v in values.items()
    )

    gg_data = GGScaleData(
        col=VectorCol("name"),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(
            label_seq=label_seq,
            value_map=value_map,
        ),
    )
    scale = cls(
        gg_data=gg_data,
        # TODO does this need a deep copy?
        color_scale=ColorScale.viridis(),
    )
    return scale


def generate_legend_markers(
    plt: ViewPort,
    scale: GGScale,
    geom_type: GeomType,
    access_idx: Optional[List[int]] = None,
) -> List[GraphicsObject]:
    """
    TODO this is noty a public function it can be pulled out
    somewhere in gg.scales
    fine for now
    """
    result: List[GraphicsObject] = []

    discrete_kind = scale.gg_data.discrete_kind
    if isinstance(discrete_kind, GGScaleDiscrete):
        if isinstance(scale, (SizeScale, ShapeScale, ColorScaleKind, FillColorScale)):
            # TODO discrete_legend_markers can become more re_usable with a yield
            # for now its fine, clean up later
            result.extend(scale.discrete_legend_markers(plt, geom_type, access_idx))
        else:
            raise Exception("`create_legend` unsupported for this scale")

    elif isinstance(discrete_kind, GGScaleContinuous):
        if isinstance(scale, (ColorScaleKind, FillColorScale)):
            mplt = deepcopy(plt)
            mplt.y_scale = discrete_kind.data_scale

            ticks = init_ticks(
                view=mplt,
                axis_kind=AxisKind.Y,
                num_ticks=5,
                bound_scale=discrete_kind.data_scale,
                is_secondary=True,
            )

            tick_labs = tick_labels(
                mplt,
                # TODO init_tickks returns type is GraphicsObjcct can change to GOTICK
                # do general refactor
                ticks,  # type: ignore
                tick_labels_input=TickLabelsInput(
                    is_secondary=True,
                    format_fn=discrete_kind.format_continuous_label,
                ),
            )
            result.extend(tick_labs)
            result.extend(ticks)
        else:
            raise GGException("Continuous legend unsupported for this scale type")

    return result


def gen_discrete_legend(
    view: ViewPort,
    cat: GGScale,
    geom_type: GeomType,
    access_idx: Optional[List[int]] = None,
):
    if not isinstance(cat.gg_data.discrete_kind, GGScaleDiscrete):
        raise GGException("expected a discrete scale")

    markers = generate_legend_markers(view, cat, geom_type, access_idx)
    num_elems = len(cat.gg_data.discrete_kind.value_map)

    layout(
        view=view,
        cols=2,
        rows=2,
        col_widths=[
            Quantity.centimeters(0.5),  # space to plot
            Quantity.relative(0.0),  # for legend including header
        ],
        row_heights=[
            Quantity.centimeters(1.0),  # for header
            Quantity.centimeters(1.05 * float(num_elems)),  # for actual legend
        ],
    )

    view.height = Quantity.centimeters(1.0 + 1.05 * float(num_elems))
    leg = view.children[3]  # Get the legend viewport

    row_heights = [Quantity.centimeters(1.05) for _ in range(num_elems)]

    layout(
        leg,
        cols=3,
        rows=num_elems,
        col_widths=[
            Quantity.centimeters(1.0),
            Quantity.centimeters(0.3),
            Quantity.relative(0.0),
        ],
        row_heights=row_heights,
    )

    j = 0  # TODO use enumerate
    for i in range(0, len(leg.children), 3):
        leg_box = leg.children[i]
        leg_label = leg.children[i + 2]

        style = Style(
            line_type=LineType.SOLID,
            line_width=1.0,
            # TODO double check alpha 1.0? not explicitly passed in i assumed so
            color=Color(1.0, 1.0, 1.0, 1.0),
            fill_color=GREY92,
        )

        y = RelativeCoordType(0.0) + init_coord_1d_from_view(
            view=leg_box, at=0.025, axis_kind=AxisKind.Y, kind=UnitType.CENTIMETER
        )
        rect = init_rect(
            leg_box,
            origin=Coord(x=RelativeCoordType(0.0), y=y),
            width=Quantity.centimeters(1.0),
            height=Quantity.centimeters(1.0),
            init_rect_input=InitRectInput(
                style=style,
                name="markerRectangle",
            ),
        )

        current_marker = markers[j]
        #  TODO double check, why was this logic here, market is already a GOText
        # so this looks redundant
        # if not isinstance(current_marker, GOPoint):
        #     # TODO this needs some refactoring on the type that is being returned
        #     raise GGException(f"expected GOPoint found {current_marker} ")

        # point = init_point_from_coord(
        #     pos=Coord(x=RelativeCoordType(0.5), y=RelativeCoordType(0.5)),
        #     marker=current_marker.marker,
        #     size=current_marker.size,
        #     color=current_marker.color,
        #     name="markerPoint",
        # )

        if isinstance(cat, (ColorScaleKind, FillColorScale, ShapeScale, SizeScale)):
            label_text = markers[j].name
        else:
            raise Exception("createLegend unsupported for this scale")

        init_text_input = InitTextInput(
            text=label_text,
            # TODO this is not accepted as an arg on our side
            # text_kind="text",
            align_kind=TextAlignKind.LEFT,
            name="markerText",
        )
        label = init_text(
            view=leg_label,
            origin=Coord(x=RelativeCoordType(0.0), y=RelativeCoordType(0.5)),
            init_text_data=init_text_input,
        )

        leg_box.add_obj(rect)
        leg_box.add_obj(current_marker)
        leg_label.add_obj(label)
        leg[i] = leg_box
        leg[i + 2] = leg_label
        j += 1

    view[3] = leg


def gen_continuous_legend(
    view: ViewPort,
    scale: GGScale,
    geom_type: GeomType,
    access_idx: Optional[List[int]] = None,
) -> None:
    """
    this could go on _ColorScaleMixin
    but for now its fine, will clean up all non public functions later
    """
    if scale.scale_type == ScaleType.SIZE:
        layout(view, cols=1, rows=6, col_widths=[], row_heights=[])

    elif isinstance(scale, (ColorScaleKind, FillColorScale)):
        # TODO this needs to come from theme.base_scale with default 1.0
        base_scale = 1.0
        # use theme.continuous_legend_height or 4.5
        height = 4.5
        # use theme.legend_header_height or 1.0
        legend_header_height = 1.0
        # use theme.continuous_legend_width
        width = 1.0

        discrete_kind = scale.gg_data.discrete_kind
        if not isinstance(discrete_kind, GGScaleContinuous):
            raise GGException("expected continuous scales")

        layout(
            view=view,
            rows=2,
            cols=2,
            col_widths=[
                Quantity.centimeters(0.5 * base_scale),
                Quantity.relative(0.0),
            ],
            row_heights=[
                Quantity.centimeters(legend_header_height),
                Quantity.centimeters(height),
            ],
        )

        leg_view = view.children[3]
        leg_view.y_scale = discrete_kind.data_scale
        layout(
            view=leg_view,
            cols=3,
            rows=1,
            col_widths=[
                Quantity.centimeters(width * base_scale),
                Quantity.centimeters(0.5 * base_scale),
                Quantity.relative(0.0),
            ],
        )

        leg_grad = leg_view.children[0]

        markers = generate_legend_markers(leg_grad, scale, geom_type, access_idx)
        for marker in markers:
            leg_grad.add_obj(marker)

        cmap = scale.color_scale
        colors = [int_to_color(color) for color in cmap.colors]
        gradient = Gradient(colors=colors)

        grad_rect = init_rect(
            leg_grad,
            origin=Coord(
                x=RelativeCoordType(0.0),
                y=RelativeCoordType(0.0),
            ),
            width=Quantity.relative(1.0),
            height=Quantity.relative(1.0),
            init_rect_input=InitRectInput(
                name="legend_gradient_background", gradient=gradient
            ),
        )

        leg_grad.add_obj(grad_rect)
        leg_view[0] = leg_grad
        view[3] = leg_view
        view.height = Quantity.centimeters(legend_header_height + height)


def create_legend(
    view: ViewPort,
    cat: GGScale,
    geom_type: GeomType,
    access_idx: Optional[List[int]] = None,
):
    start_idx = len(view.children)

    if cat.is_discrete():
        gen_discrete_legend(view, cat, geom_type, access_idx)
    elif cat.is_continuous():
        gen_continuous_legend(view, cat, geom_type, access_idx)
    else:
        raise GGException("unexpected discrete type")

    if start_idx < len(view.children):
        header = view.children[1]
        # TODO: add support to change font of legend
        label = init_text(
            header,
            # TODO sanity check this
            # nim version uses y relative(0.5)
            # both seem to be created with 1cm height
            # so probably something else causing an issue here
            # this is fine for now, but most likely to cause using when using themes
            origin=Coord(x=RelativeCoordType(0.0), y=RelativeCoordType(0.1)),
            init_text_data=InitTextInput(
                text=str(cat.gg_data.col),
                align_kind=TextAlignKind.LEFT,
                name="legend_header",
                font=Font(size=12.0, bold=True),
            ),
        )
        label.data.font.bold = True
        header.add_obj(label)


def finalize_legend(view: ViewPort, legends: List[ViewPort]):
    row_heights = [Quantity.relative(0.0)]

    for legend in legends:
        row_heights.append(legend.height)

    row_heights.append(Quantity.relative(0.0))

    layout(
        view=view,
        cols=1,
        rows=len(row_heights),
        row_heights=row_heights,
        ignore_overflow=True,
    )

    for i in range(1, len(row_heights) - 1):
        legend = legends[i - 1]
        legend.origin = view.children[i].origin
        view[i] = legend


def legend_position(x: float = 0.0, y: float = 0.0):
    return Theme(legend_position=Coord(x=RelativeCoordType(x), y=RelativeCoordType(y)))


def legend_order(idx: List[int]) -> Theme:
    return Theme(legend_order=idx)


def hide_legend() -> Theme:
    return Theme(hide_legend=True)


def canvas_color(color: PossibleColor) -> Theme:
    return Theme(canvas_color=color)


def grid_lines(
    enable: bool = True,
    # TODO medium priority
    # i dont like this inf here id rather use None, but it may originate from calculations
    # keep for now, but revisit
    width: float = float("inf"),
    color: str = "white",
    only_axes: bool = False,
) -> Theme:
    theme = Theme(grid_lines=enable, grid_line_color=color, only_axes=only_axes)

    if not math.isinf(width):
        theme.grid_line_width = width

    return theme


def minor_grid_lines(
    enable: bool = True,
    # TODO same as _grid_lines refactor inf
    width: float = float("inf"),
) -> Theme:
    theme = Theme(minor_grid_lines=enable)

    if not math.isinf(width):
        theme.minor_grid_line_width = width

    return theme


def background_color(color: PossibleColor = None) -> Theme:
    if color is None:
        color = GREY92
    return Theme(plot_background_color=color)


def grid_line_color(color: PossibleColor = None) -> Theme:
    """
    deprecated, use grid_lines
    """
    if color is None:
        color = WHITE
    return Theme(grid_line_color=color)


def parse_text_align_string(
    align_to: Literal["none", "left", "right", "center"],
) -> Optional[TextAlignKind]:
    if align_to == "none":
        return None
    return TextAlignKind(align_to)


def x_margin(
    margin: Union[int, float], outside_range: OUTSIDE_RANGE_KIND_VALUES = "none"
) -> Theme:
    if float(margin) < 0.0:
        raise GGException(
            "Margins must be positive! To make the plot range smaller use `xlim`!"
        )

    if outside_range == "none":
        or_opt = None
    else:
        or_opt = OutsideRangeKind(outside_range)

    return Theme(x_margin=float(margin), x_outside_range=or_opt)


def y_margin(
    margin: Union[int, float], outside_range: OUTSIDE_RANGE_KIND_VALUES = "none"
) -> Theme:
    if float(margin) < 0.0:
        raise GGException(
            "Margins must be positive! To make the plot range smaller use `ylim`!"
        )

    if outside_range == "none":
        or_opt = None
    else:
        or_opt = OutsideRangeKind(outside_range)

    return Theme(y_margin=float(margin), y_outside_range=or_opt)


def margin(
    left: Optional[float] = None,
    right: Optional[float] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    unit: Union[UNIT_TYPE_VALUES, UnitType] = UnitType.CENTIMETER,
) -> Theme:
    """
    TODO low priority
    there was some logic here eg px->point cm->cetnimeters from str
    we keep it simple for now, good to have later
    """
    if isinstance(unit, str):
        unit = UnitType(unit)

    return Theme(
        plot_margin_left=Quantity.from_type_or_none(unit, left),
        plot_margin_right=Quantity.from_type_or_none(unit, right),
        plot_margin_top=Quantity.from_type_or_none(unit, top),
        plot_margin_bottom=Quantity.from_type_or_none(unit, bottom),
    )


def facet_margin(
    margin: Union[Quantity, float, int], unit_type: UnitType = UnitType.CENTIMETER
) -> Theme:
    if isinstance(margin, (int, float)):
        margin = Quantity.from_type(unit_type, float(margin))

    return Theme(facet_margin=margin)


def annotate(
    text: str,
    left: Optional[float] = None,
    bottom: Optional[float] = None,
    right: Optional[float] = None,
    top: Optional[float] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    size: int = 12,
    rotate: float = 0.0,
    background_color: str = "white",
) -> Annotation:

    bg_color = to_opt_color(background_color)
    if bg_color is None:
        # TODO: implement hex (str) -> Color
        raise GGException(f"coulnd not convert {background_color} to color")

    result = Annotation(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        x=x,
        y=y,
        text=text,
        font=Font(size=size),
        rotate=rotate,
        background_color=bg_color,
    )

    if (result.x is None and result.left is None) or (
        result.y is None and result.bottom is None
    ):
        raise ValueError(
            "Both an x/left and y/bottom position has to be given to `annotate`!"
        )

    return result


def apply_theme(plt_theme: Theme, theme: Theme) -> None:
    for field in theme.__dataclass_fields__:
        value = getattr(theme, field)
        if value:
            setattr(plt_theme, field, value)


def apply_scale(aes: Aesthetics, scale: GGScale):
    result = deepcopy(aes)

    def assign_copy_scale(aes_: Aesthetics, field: str):

        if hasattr(aes_, field) and getattr(aes, field) is not None:
            new_scale = deepcopy(scale)
            field_value = getattr(aes, field)

            new_scale.gg_data.col = field_value.gg_data.col
            new_scale.gg_data.ids = field_value.gg_data.ids

            setattr(aes_, field, new_scale)

        return aes_

    SCALE_TYPE_TO_FIELD = {
        ScaleType.COLOR: "color",
        ScaleType.FILL_COLOR: "fill",
        ScaleType.ALPHA: "alpha",
        ScaleType.SIZE: "size",
        ScaleType.SHAPE: "shape",
        ScaleType.TEXT: "text",
    }

    AXIS_KIND_FIELDS = {
        AxisKind.X: ["x", "x_min", "x_max"],
        AxisKind.Y: ["y", "y_min", "y_max"],
    }

    if isinstance(scale, (LinearDataScale, TransformedDataScale)):
        if scale.data is not None:
            for field_ in AXIS_KIND_FIELDS.get(scale.data.axis_kind, []):
                result = assign_copy_scale(result, field_)
    else:
        field = SCALE_TYPE_TO_FIELD.get(scale.scale_type)
        if field is not None:
            result = assign_copy_scale(result, field)

    return result


def any_scale(arg: Any) -> bool:
    # TODO CRITICAL
    # there is a known bug here
    # there is some funny logic outside of this that does main/more that is broke
    return bool(arg.main is not None or len(arg.more) > 0)


def requires_legend(filled_scales: FilledScales, theme: Theme):
    """
    TODO CRITICAL
    there's a known bug here
    need to figure out exactly the usage of arg.main vs arg.more
    fine for now
    template anyScale(arg: untyped): untyped =
      if arg.main.isSome or arg.more.len > 0:
        true
      else:
        false
    """
    if theme.hide_legend is None and (
        filled_scales.color
        or filled_scales.fill
        or filled_scales.size
        or filled_scales.shape
    ):
        return True

    return False


def init_theme_margin_layout(
    theme: Theme, tight_layout: bool, requires_legend: bool
) -> ThemeMarginLayout:
    if theme.plot_margin_left is not None:
        left = theme.plot_margin_left
    elif tight_layout:
        left = Quantity.centimeters(0.2)
    else:
        left = Quantity.centimeters(2.5)

    if theme.plot_margin_right is not None:
        right = theme.plot_margin_right
    elif requires_legend:
        right = Quantity.centimeters(5.0)
    else:
        right = Quantity.centimeters(1.0)

    if theme.plot_margin_top is not None:
        top = theme.plot_margin_top
    elif requires_legend:
        top = Quantity.centimeters(1.25)
    else:
        top = Quantity.centimeters(1.0)

    if theme.plot_margin_bottom is not None:
        bottom = theme.plot_margin_bottom
    else:
        bottom = Quantity.centimeters(2.0)

    return ThemeMarginLayout(
        left=left, right=right, top=top, bottom=bottom, requires_legend=requires_legend
    )


def plot_layout(view: ViewPort, theme_layout: ThemeMarginLayout):

    layout(
        view=view,
        cols=3,
        rows=3,
        col_widths=[theme_layout.left, Quantity.relative(0.0), theme_layout.right],
        row_heights=[theme_layout.top, Quantity.relative(0.0), theme_layout.bottom],
    )

    view.children[0].name = "top_left"
    view.children[1].name = "title"
    view.children[2].name = "top_right"
    view.children[3].name = "y_label"
    view.children[4].name = "plot"
    view.children[5].name = "legend" if theme_layout.requires_legend else "no_legend"
    view.children[6].name = "bottom_left"
    view.children[7].name = "x_label"
    view.children[8].name = "bottom_right"


def create_layout(view: ViewPort, filled_scales: FilledScales, theme: Theme):
    hide_ticks = theme.hide_ticks or False
    hide_labels = theme.hide_labels or False
    tight_layout = hide_labels and hide_ticks
    requires_legend_ = requires_legend(filled_scales, theme)
    layout = init_theme_margin_layout(theme, tight_layout, requires_legend_)
    plot_layout(view, layout)


def handle_grid_lines(
    view: ViewPort,
    xticks: List[GraphicsObject],
    yticks: List[GraphicsObject],
    theme: Theme,
) -> List[GraphicsObject]:
    grid_line_style = get_grid_line_style(theme)
    result: List[GraphicsObject] = []

    if theme.grid_lines is None or theme.grid_lines:  # None or True
        result = [
            init_grid_lines(
                # TODO medium priority easy task
                # we get graphics object but func expects GOTick (subclass)
                # the GO related code needs to be cleaned up to take and give the concrete classes
                x_ticks=xticks,  # type: ignore
                y_ticks=yticks,  # type: ignore
                style=grid_line_style,
            )
        ]
    elif theme.only_axes:
        # only draw axes with grid line style
        result = [
            view.x_axis(grid_line_style.line_width, grid_line_style.color),
            view.y_axis(grid_line_style.line_width, grid_line_style.color),
        ]

    if theme.minor_grid_lines:
        minor_grid_line_style = get_minor_grid_line_style(grid_line_style, theme)
        result.append(
            init_grid_lines(
                # FIX / refactor same as above, fine for now
                x_ticks=xticks,  # type: ignore
                y_ticks=yticks,  # type: ignore
                major=False,
                style=minor_grid_line_style,
            )
        )

    return result


@no_type_check  # This factor needs complete re-write, just ignore types for now
def handle_labels(view: ViewPort, theme: Theme):
    """
    TODO needs a good amount of refactor
    """
    x_lab_obj = None
    y_lab_obj = None
    x_margin = None
    y_margin = None

    x_lab_txt = theme.x_label
    y_lab_txt = theme.y_label

    # ignore the types, since this will be refactored fairly soon
    @no_type_check
    def get_margin(theme_field: Any, name_val: str, ax_kind: AxisKind):
        if not theme_field is not None:
            labs = [obj for obj in view.objects if obj.name == name_val]

            # TODO medium priority easy task
            # refactor this, making it functional first
            classes = {i.__class__ for i in labs}

            if len(labs) == 0:
                raise GGException("GO object not found")

            if not classes.issubset({GOLabel, GOText, GOTickLabel}):
                raise GGException(f"expected text GO obj recived {classes}")

            lab_names = [lab.data.text for lab in labs]  # type: ignore WORKS for now but needs fixing
            lab_lengths = [len(lab_name) for lab_name in lab_names]  # type: ignore
            lab_max_length = max(lab_lengths)

            # TODO medium priority, medium complexity
            # what if theres 2 of the same length?
            max_item = [i for i in lab_names if len(i) == lab_max_length][0]  # type: ignore

            font = theme.label_font or Font(size=8.0)

            if ax_kind == AxisKind.X:
                return StrHeightCoordType(
                    pos=1.0,
                    data=TextCoordData(text=max_item, font=font),  # type: ignore
                ) + CentimeterCoordType(pos=0.3, data=LengthCoord())
            elif ax_kind == AxisKind.Y:
                return StrWidthCoordType(
                    pos=1.0,
                    data=TextCoordData(text=max_item, font=font),  # type: ignore
                ) + CentimeterCoordType(pos=0.3, data=LengthCoord())
            else:
                raise GGException("Unexpected axis")
        else:
            return CentimeterCoordType(pos=theme_field, data=LengthCoord())

    # ignore the types, since this will be refactored fairly soon
    @no_type_check
    def create_label(
        lab_proc, lab_txt, theme_field, margin_val, is_second=False, rot=None
    ):
        fnt = theme.label_font or Font()

        if theme_field:
            return lab_proc(
                view,
                lab_txt,
                margin=margin_val,
                is_custom_margin=True,
                is_secondary=is_second,
                font=fnt,
            )
        else:
            return lab_proc(
                view, lab_txt, margin=margin_val, is_secondary=is_second, font=fnt
            )

    x_margin = get_margin(theme.x_label_margin, "x_tick_label", AxisKind.X)
    y_margin = get_margin(theme.y_label_margin, "y_tick_label", AxisKind.Y)

    y_lab_obj = create_label(ylabel, y_lab_txt, theme.y_label_margin, y_margin)
    x_lab_obj = create_label(xlabel, x_lab_txt, theme.x_label_margin, x_margin)
    view.add_obj(x_lab_obj)
    view.add_obj(y_lab_obj)

    # Handle secondary axes if present
    if has_secondary(theme, AxisKind.X):
        sec_axis_label = theme.x_label_secondary
        x_margin = get_margin(theme.x_label_margin, "x_tick_label_secondary", "x")
        lab_sec = create_label(
            xlabel, sec_axis_label, theme.y_label_margin, x_margin, True
        )
        view.add_obj(lab_sec)

    if has_secondary(theme, AxisKind.Y):
        sec_axis_label = theme.y_label_secondary
        y_margin = get_margin(theme.y_label_margin, "y_tick_label_secondary", "y")
        lab_sec = create_label(
            ylabel, sec_axis_label, theme.y_label_margin, y_margin, True
        )
        view.add_obj(lab_sec)


def calc_ridge_view_map(
    ridge: Ridges, label_seq: List[GGValue]
) -> Tuple[Dict[GGValue, int], List[GGValue]]:
    num_labels = len(label_seq)
    result: Dict[GGValue, int] = {}

    if len(ridge.label_order) == 0:
        for i, label in enumerate(label_seq):
            result[label] = i + 1
    else:
        label_seq.clear()

        pair_idx = sorted(
            [(label, idx) for label, idx in ridge.label_order.items()],
            key=lambda x: x[1],
        )

        for label, idx in pair_idx:
            if idx >= num_labels:
                raise GGException(
                    f"Given `label_order` indices must not exceed the "
                    f"number of labels! Max index: {idx}, number of labels: {num_labels}"
                )
            result[label] = idx + 1
            label_seq.append(label)

    return result, label_seq


def create_ridge_layout(view: ViewPort, theme: Theme, num_labels: int):
    if theme.discrete_scale_margin is not None:
        discrete_margin = theme.discrete_scale_margin
    else:
        discrete_margin = Quantity.relative(0.0)

    ind_heights = [Quantity.relative(0.0) for _ in range(num_labels)]

    layout(
        view,
        cols=1,
        rows=num_labels + 2,
        row_heights=[discrete_margin] + ind_heights + [discrete_margin],
        ignore_overflow=True,
    )


# TODO Refactor this
def generate_ridge(
    view: ViewPort,
    ridge: Ridges,
    p: GgPlot,
    filled_scales: FilledScales,
    theme: Theme,
    hide_labels: bool = False,
    hide_ticks: bool = False,
):
    y_ridge_scale = filled_scales.y_ridges.main

    if not isinstance(y_ridge_scale.gg_data.discrete_kind, GGScaleDiscrete):
        raise GGException("expected discrete scale")

    y_label_seq = y_ridge_scale.gg_data.discrete_kind.label_seq
    num_labels = len(y_label_seq)

    y_scale = theme.y_range or filled_scales.y_scale
    y_scale = Scale(low=y_scale.low, high=y_scale.high / ridge.overlap)  # type: ignore
    view.y_scale = y_scale

    view_map, y_label_seq = calc_ridge_view_map(ridge, y_label_seq)
    create_ridge_layout(view, theme, num_labels)

    view_map_items = view_map.items()
    # for ridges we have to draw in reverse
    # so that the item that overlaps is drawn above
    # i think this logic is correct, need to double check
    view_map_items = sorted(view_map_items, key=lambda x: -x[1])
    for label, idx in view_map_items:
        view_label = view.children[idx]
        for cnt, fg in enumerate(filled_scales.geoms):
            p_child = view_label.add_viewport_from_coords(
                CoordsInput(), ViewPortInput(name=f"data_{cnt}")
            )

            # Create theme which ignores points outside the scale
            m_theme = deepcopy(theme)
            m_theme.x_outside_range = OutsideRangeKind.NONE
            m_theme.y_outside_range = OutsideRangeKind.NONE

            create_gobj_from_geom(
                p_child, fg, m_theme, label_val={"col": str(ridge.col), "val": label}
            )

            # Add data viewport to the view
            view_label.children.append(p_child)

        if ridge.show_ticks:
            # TODO: fix the hack using 1e-5. Needed for side effect of adding to viewport
            handle_ticks(
                view_label,
                filled_scales,
                p,
                AxisKind.Y,
                theme=theme,
                num_ticks_opt=5,
                bound_scale_opt=Scale(low=y_scale.low + 1e-5, high=y_scale.high - 1e-5),
            )

    if not hide_ticks:
        x_ticks = handle_ticks(view, filled_scales, p, AxisKind.X, theme=theme)

        format_func = (
            y_ridge_scale.gg_data.discrete_kind.format_discrete_label
            if y_ridge_scale.gg_data.discrete_kind.format_discrete_label is not None
            else str
        )

        # Create ticks manually with discrete_tick_labels to set labels
        y_ticks = handle_discrete_ticks(
            view,
            p,
            AxisKind.Y,
            y_label_seq,
            theme=theme,
            center_ticks=False,
            format_func=format_func,
        )

        grid_lines = handle_grid_lines(view, x_ticks, y_ticks, theme)
        for grid_line in grid_lines:
            view.add_obj(grid_line)

    if not hide_labels:
        handle_labels(view, theme)

    view.x_scale = theme.x_margin_range

    # Handle axis reversals
    if not filled_scales.discrete_x and filled_scales.reversed_x:

        view.x_scale = Scale(low=view.x_scale.high, high=view.x_scale.low)

    if not filled_scales.discrete_y and filled_scales.reversed_y:
        view.y_scale = Scale(low=view.y_scale.high, high=view.y_scale.low)


def _generate_plot_geoms(view: ViewPort, filled_scales: FilledScales, theme: Theme):
    for cnt, geom in enumerate(filled_scales.geoms):
        coords_input = CoordsInput()
        viewport_input = ViewPortInput(name=f"data_{cnt}")
        p_child = view.add_viewport_from_coords(coords_input, viewport_input)

        create_gobj_from_geom(p_child, geom, theme)
        view.children.append(p_child)


def _generate_plot_ticks(
    view: ViewPort,
    filled_scales: FilledScales,
    plot: GgPlot,
    theme: Theme,
    hide_ticks: bool,
) -> Tuple[List[GraphicsObject], List[GraphicsObject]]:
    # TODO this needs to be moved out of public interface eventually
    x_ticks: List[GraphicsObject] = []
    y_ticks: List[GraphicsObject] = []
    if not hide_ticks:
        # TODO double check num_ticks_opt=10
        x_ticks = handle_ticks(
            view, filled_scales, plot, AxisKind.X, num_ticks_opt=10, theme=theme
        )
        y_ticks = handle_ticks(
            view, filled_scales, plot, AxisKind.Y, num_ticks_opt=10, theme=theme
        )
    return x_ticks, y_ticks


def _generate_plot_update_scales(
    view: ViewPort,
    filled_scales: FilledScales,
    x_ticks: List[GraphicsObject],
    y_ticks: List[GraphicsObject],
    theme: Theme,
    hide_ticks: bool,
):
    # TODO this needs to be moved out of public interface eventually
    view.x_scale = theme.x_margin_range
    view.y_scale = theme.y_margin_range

    if not filled_scales.discrete_x and filled_scales.reversed_x:
        view.x_scale = Scale(low=view.x_scale.high, high=view.x_scale.low)
    if not filled_scales.discrete_y and filled_scales.reversed_y:
        view.y_scale = Scale(low=view.y_scale.high, high=view.y_scale.low)

    view.update_data_scale()
    if not hide_ticks:
        view.update_data_scale_for_objects(x_ticks)
        view.update_data_scale_for_objects(y_ticks)


# TODO refactor
def generate_plot(
    view: ViewPort,
    plot: GgPlot,
    filled_scales: FilledScales,
    theme: Theme,
    hide_labels: bool = False,
    hide_ticks: bool = False,
):
    background_style = get_plot_background(theme)
    background(view, style=background_style)

    view.x_scale = theme.x_range or filled_scales.x_scale

    if plot.ridges is not None:
        ridge = plot.ridges
        generate_ridge(view, ridge, plot, filled_scales, theme, hide_labels, hide_ticks)
    else:
        view.y_scale = theme.y_range = theme.y_range or filled_scales.y_scale

        _generate_plot_geoms(view, filled_scales, theme)

        x_ticks, y_ticks = _generate_plot_ticks(
            view, filled_scales, plot, theme, hide_ticks
        )

        _generate_plot_update_scales(
            view,
            filled_scales,
            x_ticks,
            y_ticks,
            theme,
            hide_ticks,
        )
        grid_lines = handle_grid_lines(view, x_ticks, y_ticks, theme)

        if not hide_labels:
            handle_labels(view, theme)

        for grid_line in grid_lines:
            view.add_obj(grid_line)


# TODO refactor
def determine_existing_combinations(fs: FilledScales, facet: Facet) -> Set[GGValue]:
    facets = fs.facets
    if len(facets) <= 0:
        raise GGException("expected facets")

    # here the facets have to be on discrete scales
    # we can assume that but would be good to structure it better
    if len(facet.columns) > 1:
        combinations = list(
            product(*(facet.gg_data.discrete_kind.label_seq for facet in facets))
        )
    else:
        combinations: List[List[GGValue]] = [[label] for label in facets[0].gg_data.discrete_kind.label_seq]  # type: ignore

    comb_labels: Set[Tuple[str, GGValue]] = set()

    for c in combinations:
        comb = []
        for i, fc in enumerate(facet.columns):
            if isinstance(c[i], np.generic):
                item = c[i].item()
            else:
                item = c[i]
            comb.append((fc, item))
        comb = tuple(comb)
        comb_labels.add(comb)

    result: Set[GGValue] = set()

    # This logic is a bit funny
    # for drv. cyl in cars we get combos in terms of:
    # (6, r), (5, f)
    # yield_data.keys() in case of aes(color = "manufacturer")
    # will return data in the form of (audio, 6, r)
    # we should probably just take all the combos from the df in yield data
    # something like df[facet.columns].drop_duplicates()
    # this should be enough, and more simple than current impl
    for fg in fs.geoms:
        for cb in comb_labels:
            for xk in fg.gg_data.yield_data.keys():
                left = {i[1] for i in cb}

                if left.issubset(set(xk)):
                    result.add(cb)

    if len(result) > len(combinations):
        raise GGException("result should be less than combinations")

    return sorted(result)


def calc_facet_view_map(comb_labels: Set[GGValue]) -> Dict[GGValue, int]:
    result: Dict[GGValue, int] = {}
    for idx, cb in enumerate(comb_labels):
        result[cb] = idx
    return result


def find_gg_value(fg: FilledGeom, label: GGValue) -> pd.DataFrame:
    result = pd.DataFrame()

    for key, val in fg.gg_data.yield_data.items():
        if label == key:
            result = pd.concat([result, val[2]], ignore_index=True)

    if len(result) <= 0:
        raise GGException("invalid call to find label")
    return result


# todo refactor
def calc_scales_for_label(theme: Theme, facet: Facet, fg: FilledGeom, label: GGValue):

    def calc_scale(df: pd.DataFrame, col: str) -> Scale:
        data = df[col].to_numpy()  # type: ignore
        return Scale(
            low=float(np.nanmin(data)),  # type: ignore
            high=float(np.nanmax(data)),  # type: ignore
        )

    if facet.scale_free_kind in {
        ScaleFreeKind.FREE_X,
        ScaleFreeKind.FREE_Y,
        ScaleFreeKind.FREE,
    }:
        lab_df = find_gg_value(fg, label)

        if facet.scale_free_kind in {ScaleFreeKind.FREE_X, ScaleFreeKind.FREE}:
            x_scale = calc_scale(lab_df, fg.gg_data.x_col)

            if x_scale.low != x_scale.high:
                theme.x_margin_range = calculate_margin_range(
                    theme, x_scale, AxisKind.X
                )
            else:
                # base on filled geom's scale instead
                theme.x_margin_range = calculate_margin_range(
                    theme, fg.gg_data.x_scale, AxisKind.X
                )

        # TODO i think this logic is wrong
        if facet.scale_free_kind in {ScaleFreeKind.FREE_Y, ScaleFreeKind.FREE}:
            y_scale = calc_scale(lab_df, fg.gg_data.y_col)
            if y_scale.low != y_scale.high:
                theme.y_margin_range = calculate_margin_range(
                    theme, y_scale, AxisKind.Y
                )
            else:
                # base on filled geom's scale instead
                theme.y_margin_range = calculate_margin_range(
                    theme, fg.gg_data.y_scale, AxisKind.Y
                )


# TODO refactor...
def generate_facet_plots(
    view: ViewPort,
    plot: GgPlot,
    filled_scales: FilledScales,
    hide_labels: bool = False,
    hide_ticks: bool = False,
):
    if plot.facet is None:
        raise GGException("facet is none..")

    facet = plot.facet

    plot.theme.x_margin = 0.05
    plot.theme.y_margin = 0.05

    theme = build_theme(filled_scales, plot)

    if theme.x_tick_label_margin is None:
        theme.x_tick_label_margin = 1.75
    if theme.y_tick_label_margin is None:
        theme.y_tick_label_margin = -1.25

    theme.x_ticks_rotate = plot.theme.x_ticks_rotate
    theme.y_ticks_rotate = plot.theme.y_ticks_rotate
    theme.x_ticks_text_align = plot.theme.x_ticks_text_align
    theme.y_ticks_text_align = plot.theme.y_ticks_text_align

    # Calculate existing combinations
    exist_comb = determine_existing_combinations(filled_scales, facet)
    num_exist = len(exist_comb)

    # Calculate rows and columns
    if theme.prefer_rows_over_columns:
        cols, rows = calc_rows_columns(0, 0, num_exist)
    else:
        rows, cols = calc_rows_columns(0, 0, num_exist)

    view_map = calc_facet_view_map(exist_comb)

    if facet.scale_free_kind in {
        ScaleFreeKind.FREE_X,
        ScaleFreeKind.FREE_Y,
        ScaleFreeKind.FREE,
    }:
        margin = theme.facet_margin or Quantity.relative(0.015)
    else:
        margin = theme.facet_margin or Quantity.relative(0.001)

    layout(view, cols=cols, rows=rows, margin=margin)

    x_ticks = []
    y_ticks = []
    last_col = num_exist % cols

    for label, idx in view_map.items():
        view_label = view.children[idx]

        layout(
            view_label,
            rows=2,
            cols=1,
            row_heights=[Quantity.relative(0.1), Quantity.relative(0.9)],
            margin=Quantity.relative(0.01),
        )

        header_view = view_label.children[0]
        background(header_view)

        text = str(label)
        header_text = init_text(
            header_view,
            origin=Coord(x=RelativeCoordType(0.5), y=RelativeCoordType(0.5)),
            init_text_data=InitTextInput(
                text=text,
                align_kind=TextAlignKind.CENTER,
                font=Font(size=8.0),
                name="facet_header_text",
            ),
        )
        header_view.add_obj(header_text)
        header_view.name = "facet_header"

        plot_view = view_label.children[1]
        background(plot_view, style=get_plot_background(theme))

        cur_row = idx // cols
        cur_col = idx % cols

        hide_x_labels = not (
            facet.scale_free_kind in {ScaleFreeKind.FREE_X, ScaleFreeKind.FREE}
            or cur_row == rows - 1
            or (cur_row == rows - 2 and cur_col >= last_col and last_col > 0)
        )

        hide_y_labels = not (
            facet.scale_free_kind in {ScaleFreeKind.FREE_X, ScaleFreeKind.FREE}
            or cur_col == 0
        )

        plot_view.name = "facet_plot"
        view_label.name = f"facet_{text}"

        set_grid_and_ticks = False
        for geom in filled_scales.geoms:
            if not set_grid_and_ticks:
                # Calculate scales for current label
                calc_scales_for_label(theme, facet, geom, label)

                plot_view.x_scale = theme.x_margin_range
                plot_view.y_scale = theme.y_margin_range

                x_tick_num = 5 if (1000.0 < theme.x_margin_range.high < 1e5) else 10

                x_ticks = handle_ticks(
                    plot_view,
                    filled_scales,
                    plot,
                    AxisKind.X,
                    theme=theme,
                    num_ticks_opt=x_tick_num,
                    hide_tick_labels=hide_x_labels,
                )

                y_ticks = handle_ticks(
                    plot_view,
                    filled_scales,
                    plot,
                    AxisKind.Y,
                    theme=theme,
                    hide_tick_labels=hide_y_labels,
                )

                grid_lines = handle_grid_lines(plot_view, x_ticks, y_ticks, theme)
                for grid_line in grid_lines:
                    plot_view.add_obj(grid_line)
                set_grid_and_ticks = True

            # Create child viewport for data
            p_child = plot_view.add_viewport_from_coords(
                CoordsInput(), ViewPortInput(name="data")
            )
            label_ = {i[1] for i in label}
            create_gobj_from_geom(p_child, geom, theme, label_val=label_)
            plot_view.children.append(p_child)

        view_label.x_scale = plot_view.x_scale
        view_label.y_scale = plot_view.y_scale

        # if not view.x_scale or not view.y_scale:
        # TODO check this
        # raise GGException("expected x and y scale")

        if not filled_scales.discrete_x and filled_scales.reversed_x:
            view_label.x_scale = Scale(high=view.x_scale.low, low=view.x_scale.high)
        if not filled_scales.discrete_y and filled_scales.reversed_y:
            view_label.y_scale = Scale(high=view.y_scale.low, low=view.y_scale.high)

    if not hide_labels:
        if theme.x_label_margin is None:
            theme.x_label_margin = 1.0
        if theme.y_label_margin is None:
            theme.y_label_margin = 1.5

        handle_labels(view, theme)


def get_left_bottom(
    view: ViewPort,
    annotation: Annotation,
    total_height: PointUnit,
    max_width: PointUnit,
) -> Tuple[float, float]:
    result_left = 0.0
    result_bottom = 0.0

    if annotation.left is not None:
        result_left = (
            Quantity.relative(annotation.left).to_points(length=view.point_width()).val
        )
    elif annotation.right is not None:
        result_left = (
            Quantity.relative(annotation.right)
            .to_points(length=view.point_width())
            .subtract(max_width)
            .val
        )
    else:
        if annotation.x is None or view.x_scale is None:
            raise GGException("expected annotation.x and view.x_scale")

        result_left = (
            DataCoordType(
                pos=annotation.x,
                data=DataCoord(axis_kind=AxisKind.X, scale=view.x_scale),
            )
            .to_points(length=view.point_width())
            .pos
        )

    if annotation.bottom is not None:
        result_bottom = (
            Quantity.relative(annotation.bottom)
            .to_points(length=view.point_height())
            .val
        )
    elif annotation.top is not None:
        result_bottom = (
            Quantity.relative(annotation.top)
            .to_points(length=view.point_height())
            .subtract(total_height)
            .val
        )
    else:
        if annotation.y is None or view.y_scale is None:
            raise GGException("expected annotation.x and view.x_scale")
        result_bottom = (
            DataCoordType(
                pos=annotation.y,
                data=DataCoord(axis_kind=AxisKind.Y, scale=view.y_scale),
            )
            .to_points(length=view.point_height())
            .pos
        )

    return (result_left, result_bottom)


def get_str_width(text: str, font: Font) -> PointUnit:
    return PointUnit(
        StrWidthCoordType(
            pos=1.0,
            data=TextCoordData(text=text, font=font),
        )
        .to_points()
        .pos
    )


def str_width(val: float, font: Font) -> StrWidthCoordType:
    return StrWidthCoordType(
        pos=val,
        data=TextCoordData(text="W", font=font),
    )


def str_height(text: str, font: Font) -> Quantity:
    num_lines = len(text.split("\n"))
    return DataUnit(
        val=StrHeightCoordType(num_lines * 1.5, data=TextCoordData(text="", font=font))
        .to_points()
        .pos
    )


def draw_annotations(view: ViewPort, plot: GgPlot) -> None:
    ANNOT_RECT_MARGIN = 0.5

    for annot in plot.annotations:
        rect_style = Style(
            fill_color=annot.background_color, color=annot.background_color
        )

        margin_h = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=annot.font),
        ).to_points()

        margin_w = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=annot.font),
        ).to_points()

        total_height: PointUnit = Quantity.points(
            str_height(annot.text, annot.font).val + (margin_h.pos * 2.0),
        )  # type: ignore

        font = annot.font
        max_line = list(
            sorted(
                annot.text.split("\n"),
                key=lambda x: get_str_width(x, font).val,
            )
        )[-1]
        max_width = get_str_width(max_line, font)

        rect_width = Quantity.points(
            max_width.val + margin_w.pos * 2.0,
        )
        left, bottom = get_left_bottom(view, annot, total_height, max_width)

        rect_x = left - margin_w.pos
        rect_y = bottom - total_height.val + margin_h.pos

        annot_rect = None
        if annot.background_color != TRANSPARENT:
            annot_rect = init_rect(
                view,
                origin=Coord(
                    x=PointCoordType(pos=rect_x), y=PointCoordType(pos=rect_y)
                ),
                width=rect_width,
                height=total_height,
                init_rect_input=InitRectInput(
                    style=rect_style, rotate=annot.rotate, name="annotationBackground"
                ),
            )

        annot_text = init_multi_line_text(
            view,
            origin=Coord(
                x=Coord1D.create_point(left, view.point_width()),
                y=Coord1D.create_point(bottom, view.point_height()),
            ),
            text=annot.text,
            text_kind=GOType.TEXT,
            align_kind=TextAlignKind.LEFT,
            init_multi_line_input=InitMultiLineInput(
                rotate=annot.rotate,
                font=annot.font,
            ),
        )

        if annot_rect is not None:
            view.add_obj(annot_rect)

        for text in annot_text:
            view.add_obj(text)


def draw_title(view: ViewPort, title: str, theme: Theme, width: Quantity):
    title = str(title)  # ensure title is string
    font = theme.title_font or Font(size=16.0)

    if "\n" not in title:
        str_width = get_str_width(title, font)
        # TODO CRITICAL, easy task
        # we have to double check if the scales are comparable
        # what if one is cm and the other is inch?
        # we could normalise them or we could raise exception if not
        # or make this logic generic with __gt__
        if str_width.val > width.val:
            # rebuild and wrap
            line = ""
            m_title = ""
            for word in title.split(" "):
                line_width = get_str_width(line + word, font)
                # TODO critical easy task
                # same as the previous comparison
                if line_width.val < width.val:
                    line += word + " "
                else:
                    m_title += line + "\n"
                    line = word + " "
            m_title += line
            title = m_title
    else:
        # user is manually wrapping and responsible
        pass

    title_obj = init_multi_line_text(
        view=view,
        # TODO check if this is correct
        origin=Coord(x=RelativeCoordType(0.0), y=RelativeCoordType(0.0)),
        text=title,
        text_kind=GOType.TEXT,
        align_kind=TextAlignKind.LEFT,
        init_multi_line_input=InitMultiLineInput(font=font),
    )
    for item in title_obj:
        view.add_obj(item)


def _get_geom_for_scale(plot: GgPlot, scale: GGScale) -> Optional[Geom]:
    """
    TODO i don't like this logic, fine for now
    """
    for geom in plot.geoms:
        if geom.gg_data.gid in scale.gg_data.ids:
            return geom


def _draw_legends(
    img: ViewPort, filled_scales: FilledScales, theme: Theme, plot: GgPlot
):
    # TODO move this out of this file eventually
    drawn_legends: Set[Tuple[DiscreteType, ScaleType, GeomType]] = set()
    scale_names: Set[str] = set()
    legends: List[ViewPort] = []

    legend_view = img.get_child_by_name({"legend", "no_legend"})

    for scale in filled_scales.enumerate_scales_by_id():
        scale_geom: Optional[Geom] = _get_geom_for_scale(plot, scale)
        if scale_geom is None:
            raise GGException("expected to find a geom for the sacale")
        if (
            theme.hide_legend is None
            and scale.scale_type
            not in {ScaleType.LINEAR_DATA, ScaleType.TRANSFORMED_DATA}
            and (
                scale.gg_data.discrete_kind.discrete_type,
                scale.scale_type,
                scale_geom.geom_type,
            )
            not in drawn_legends
        ):
            lg = deepcopy(legend_view)
            create_legend(lg, scale, scale_geom.geom_type, theme.legend_order)

            # TODO low priority, high difficulty
            # support FormulaNode
            scale_col = str(scale.gg_data.col)

            if scale_col not in scale_names:
                legends.append(lg)
                drawn_legends.add(
                    (
                        scale.gg_data.discrete_kind.discrete_type,
                        scale.scale_type,
                        scale_geom.geom_type,
                    )
                )
            scale_names.add(scale_col)

    if len(legends) > 0:
        finalize_legend(legend_view, legends)
        if plot.theme.legend_position:
            pos = plot.theme.legend_position
            legend_view.origin.x = pos.x
            legend_view.origin.y = pos.y


def _draw_title(img: ViewPort, theme: Theme, plot: GgPlot):
    # todo move out of this file eventually
    if plot.title and len(plot.title) > 0:
        title_viewport = img.get_child_by_name("title")
        top_right_viewport = img.get_child_by_name("top_right")
        draw_title(
            title_viewport,
            plot.title,
            theme,
            title_viewport.point_width().add(top_right_viewport.point_width()),
        )


def _collect_scales(plot: GgPlot) -> FilledScales:
    filled_scales: FilledScales
    if plot.ridges is not None:
        filled_scales = collect_scales(plot.update_aes_ridges())
    else:
        filled_scales = collect_scales(plot)

    return filled_scales


def _generate_plot(
    plt_base: ViewPort, theme: Theme, plot: GgPlot, filled_scales: FilledScales
):
    # todo move out of this file eventually
    hide_ticks = theme.hide_ticks or False
    hide_labels = theme.hide_labels or False

    if plot.facet is not None:
        generate_facet_plots(
            plt_base,
            plot,
            filled_scales,
            hide_labels=hide_labels,
            hide_ticks=hide_ticks,
        )
    else:
        generate_plot(
            plt_base,
            plot,
            filled_scales,
            theme,
            hide_labels=hide_labels,
            hide_ticks=hide_ticks,
        )


def ggcreate(plot: GgPlot, width: float = 640.0, height: float = 480.0) -> PlotView:
    if len(plot.geoms) == 0:
        raise GGException("Please use at least one `geom` or ridges")

    filled_scales: FilledScales = _collect_scales(plot)
    post_process_scales(filled_scales, plot)
    theme = build_theme(filled_scales, plot)

    coord_input = CoordsInput()
    viewport_input = ViewPortInput(
        name="root",
        w_img=Quantity.points(width),
        h_img=Quantity.points(height),
    )
    img = ViewPort.from_coords(coord_input, viewport_input)

    background_style = get_canvas_background(theme)
    background(img, style=background_style)
    create_layout(img, filled_scales, theme)

    plt_base = img.get_child_by_name("plot")

    _generate_plot(plt_base, theme, plot, filled_scales)

    img.x_scale = plt_base.x_scale
    img.y_scale = plt_base.y_scale

    _draw_legends(img, filled_scales, theme, plot)
    draw_annotations(img.get_child_by_name("plot"), plot)
    _draw_title(img, theme, plot)

    return PlotView(filled_scales=filled_scales, view=img)


def to_tex_options():
    # after alpha version
    raise GGException("unsupported")


def fill_empty_spaces(
    rows: int,
    cols: int,
    items: List[Optional[Any]],
    horizontal_orientation: str,
    vertical_orientation: str,
) -> List[Optional[Any]]:
    total_cells = rows * cols
    items = items[:total_cells] + [None] * (total_cells - len(items))

    if "top_to_bottom" == vertical_orientation:
        row_order = list(range(rows))
    else:
        row_order = list(reversed(range(rows)))

    if "left_to_right" == horizontal_orientation:
        col_order = list(range(cols))
    else:
        col_order = list(reversed(range(cols)))

    positions = [(r, c) for r in row_order for c in col_order]

    grid = [None] * total_cells

    for item, (r, c) in zip(items, positions):
        grid[r * cols + c] = item

    return grid


def empty_view_for_gg_mmulti(img: ViewPort, h_val: float, w_val: float):
    empty_view = ViewPort.from_coords(
        CoordsInput(),
        ViewPortInput(
            w_img=PointUnit(h_val),
            h_img=PointUnit(w_val),
        ),
    )
    background_style = img.get_current_background_style()

    background(empty_view, background_style)
    return empty_view


def ggmulti(
    plts: List[GgPlot],
    fname: Union[str, Path],
    width: int = 640,
    height: int = 480,
    widths: Optional[List[int]] = None,
    heights: Optional[List[int]] = None,
    use_tex: bool = False,
    only_tikz: bool = False,
    standalone: bool = False,
    tex_template: str = "",
    caption: str = "",
    label: str = "",
    placement: str = "htbp",
    horizontal_orientation: Literal["left_to_right", "right_to_left"] = "left_to_right",
    vertical_orientation: Literal["top_to_bottom", "bottom_to_top"] = "top_to_bottom",
):
    """
    backend is fixated to cairo for now
    TODO only_tikz, use_tex not used (need to remove them for now)
    """
    if widths is None:
        widths = []
    if heights is None:
        heights = []
    width = widths[0] if len(widths) == 1 else width
    height = heights[0] if len(heights) == 1 else height

    def raise_if_not_matching(arg: Any, arg_name: Any):
        if len(arg) > 1 and len(arg) != len(plts):
            raise ValueError(
                f"Incorrect number of {arg_name} in call to ggmulti. "
                f"Has {len(arg)}, but needs: {len(plts)}"
            )

    raise_if_not_matching(widths, "widths")
    raise_if_not_matching(heights, "heights")

    if len(widths) > 0 or len(heights) > 0:
        # Use explicit widths/heights
        w_val = sum(widths) if widths else width
        h_val = sum(heights) if heights else height
        img = ViewPort.from_coords(
            CoordsInput(),
            ViewPortInput(w_img=PointUnit(w_val), h_img=PointUnit(h_val)),
        )
        widths_q: List[PointUnit] = (
            [PointUnit(float(w)) for w in widths] if widths else []
        )
        heights_q: List[PointUnit] = (
            [PointUnit(float(h)) for h in heights] if heights else []
        )
        layout(
            img,
            cols=max(len(widths), 1),
            rows=max(len(heights), 1),
            col_widths=widths_q,  # type: ignore
            row_heights=heights_q,  # type: ignore
        )

        rows, cols = len(heights_q), len(widths_q)
    else:
        rows, cols = calc_rows_columns(0, 0, len(plts))
        img = ViewPort.from_coords(
            CoordsInput(),
            ViewPortInput(
                w_img=PointUnit(width * cols),
                h_img=PointUnit(height * rows),
            ),
        )
        layout(img, cols=cols, rows=rows)

    plts_filled = fill_empty_spaces(
        rows, cols, plts, horizontal_orientation, vertical_orientation
    )

    for i, plt in enumerate(plts_filled):
        w_val = widths[i] if i < len(widths) else width
        h_val = heights[i] if i < len(heights) else height
        if plt is None:
            new_view = empty_view_for_gg_mmulti(img, h_val, w_val)
        else:
            pp = ggcreate(plt, width=w_val, height=h_val)
            new_view = pp.view
        view_embed_at(img, i, new_view)

    draw_to_file(img, fname)
