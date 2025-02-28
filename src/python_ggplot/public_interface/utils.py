"""
TODO this whole file needs cleaning up
"""

import math
from collections import OrderedDict
from copy import deepcopy
from itertools import product
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

from python_ggplot.common.enum_literals import (
    OUTSIDE_RANGE_KIND_VALUES,
    UNIT_TYPE_VALUES,
)
from python_ggplot.core.chroma import int_to_color
from python_ggplot.core.coord.objects import (
    CentimeterCoordType,
    Coord,
    CoordsInput,
    DataCoord,
    DataCoordType,
    LengthCoord,
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
from python_ggplot.core.units.objects import DataUnit, PointUnit, Quantity
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    GGValue,
    VectorCol,
    VFillColor,
    VLinearData,
)
from python_ggplot.gg.drawing import create_gobj_from_geom
from python_ggplot.gg.geom.base import FilledGeom
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
from python_ggplot.graphics.draw import background, draw_to_file, layout
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


def scale_axis_log(
    axis_kind: AxisKind, base: int, breaks: Optional[Union[int, List[float]]] = None
) -> GGScale:
    def trans(v: float) -> float:
        return BASE_TO_LOG[base](v)

    def inv_trans(v: float) -> float:
        return math.pow(base, v)

    # TODO this leaves room for errors
    gg_data = GGScaleData(
        col=VectorCol(""),  # will be filled when added to GgPlot obj
        value_kind=VTODO(),  # i guess here same with col, will be added later
        discrete_kind=GGScaleContinuous(),
    )
    scale = TransformedDataScale(
        gg_data=gg_data,
        data=LinearAndTransformScaleData(axis_kind=axis_kind),
        transform=trans,
        inverse_transform=inv_trans,
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
    plt: ViewPort, scale: GGScale, access_idx: Optional[List[int]] = None
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
            result.extend(scale.discrete_legend_markers(plt, access_idx))
        else:
            raise Exception("`create_legend` unsupported for this scale")

    elif isinstance(discrete_kind, GGScaleContinuous):
        if isinstance(scale, (ColorScaleKind, FillColorScale)):
            # TODO CRITICAL
            # examine this in detail, do we need a deepcopy?
            # why does the original code do var mplt = plt
            # we have to investigate, this can cause many bugs
            mplt = plt
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
                    margin=init_coord_1d_from_view(
                        view=plt, at=0.3, axis_kind=AxisKind.X, kind=UnitType.CENTIMETER
                    ),
                    format_fn=discrete_kind.format_continuous_label,
                ),
            )
            result.extend(tick_labs)
            result.extend(ticks)
        else:
            raise GGException("Continuous legend unsupported for this scale type")

    return result


def gen_discrete_legend(
    view: ViewPort, cat: GGScale, access_idx: Optional[List[int]] = None
):
    if not isinstance(cat.gg_data.discrete_kind, GGScaleDiscrete):
        raise GGException("expected a discrete scale")

    markers = generate_legend_markers(view, cat, access_idx)
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
        if not isinstance(current_marker, GOPoint):
            # TODO this needs some refactoring on the type that is being returned
            raise GGException("expected GOPoint or ")

        point = init_point_from_coord(
            pos=Coord(x=RelativeCoordType(0.5), y=RelativeCoordType(0.5)),
            marker=current_marker.marker,
            size=current_marker.size,
            color=current_marker.color,
            name="markerPoint",
        )

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
        leg_box.add_obj(point)
        leg_label.add_obj(label)
        leg[i] = leg_box
        leg[i + 2] = leg_label
        j += 1

    view[3] = leg


def gen_continuous_legend(
    view: ViewPort, scale: GGScale, access_idx: Optional[List[int]] = None
) -> None:
    """
    this could go on _ColorScaleMixin
    but for now its fine, will clean up all non public functions later
    """
    if scale.scale_type == ScaleType.SIZE:
        layout(view, cols=1, rows=6, col_widths=[], row_heights=[])

    elif isinstance(scale, (ColorScaleKind, FillColorScale)):
        discrete_kind = scale.gg_data.discrete_kind
        if not isinstance(discrete_kind, GGScaleContinuous):
            raise GGException("expected continuous scales")

        layout(
            view=view,
            rows=2,
            cols=2,
            col_widths=[
                Quantity.centimeters(0.5),
                Quantity.relative(0.0),
            ],
            row_heights=[
                Quantity.centimeters(1.0),
                Quantity.centimeters(4.5),
            ],
        )

        leg_view = view.children[3]
        leg_view.y_scale = discrete_kind.data_scale
        layout(
            leg_view,
            3,
            1,
            col_widths=[
                Quantity.centimeters(1.0),
                Quantity.centimeters(0.5),
                Quantity.relative(0.0),
            ],
        )

        leg_grad = leg_view.children[0]

        markers = generate_legend_markers(leg_grad, scale, access_idx)
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
                name="legendGradientBackground", gradient=gradient
            ),
        )

        leg_grad.add_obj(grad_rect)
        leg_view[0] = leg_grad
        view[3] = leg_view
        view.height = Quantity.centimeters(5.5)


def create_legend(
    view: ViewPort, cat: GGScale, access_idx: Optional[List[int]] = None
) -> None:
    # TODO high priority / easy task
    # double check this to be sure, original code is len(view)
    # i remember ginger sets this up, its either len(view.objects) or  len(view.children)
    start_idx = len(view.children)

    if cat.is_discrete():
        gen_discrete_legend(view, cat, access_idx)
    elif cat.is_continuous():
        gen_continuous_legend(view, cat, access_idx)
    else:
        raise GGException("unexpected discrete type")

    if start_idx < len(view.children):
        header = view.children[1]
        # TODO: add support to change font of legend
        label = init_text(
            header,
            origin=Coord(x=RelativeCoordType(0.0), y=RelativeCoordType(0.5)),
            init_text_data=InitTextInput(
                # TODO sanity check this, original code calls evaluate (for FormulaNode)
                # make sure we are fine with this, certainly not calling eval
                text=str(cat.gg_data.col),
                align_kind=TextAlignKind.LEFT,
                name="legendHeader",
            ),
        )
        label.data.font.bold = True
        header.add_obj(label)
        view[1] = header


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


def background_color(color: PossibleColor = GREY92) -> Theme:
    return Theme(plot_background_color=color)


def grid_line_color(color: PossibleColor = WHITE) -> Theme:
    """
    deprecated, use grid_lines
    """
    return Theme(grid_line_color=color)


def parse_text_align_string(
    align_to: Literal["none", "left", "right", "center"]
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
    x: Optional[float] = None,
    y: Optional[float] = None,
    font: Font = Font(size=12.0),
    rotate: float = 0.0,
    background_color: str = "white",
) -> Annotation:

    if background_color == "white":
        bg_color = WHITE
    else:
        # TODO CRITICAL EASY TASK
        # there is a function for str -> color in chroma
        # this is pending to be ported and it has its own TODO
        # that work blocks this code here, but dont want to sidetrack for now
        raise GGException("needs to be implemented")

    result = Annotation(
        left=left,
        bottom=bottom,
        x=x,
        y=y,
        text=text,
        font=font,
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
        if value is not None:
            setattr(plt_theme, field, value)


def apply_scale(aes: Aesthetics, scale: GGScale):
    result = deepcopy(aes)

    def assign_copy_scale(field: str):
        if hasattr(aes, field) and getattr(aes, field) is not None:
            new_scale = deepcopy(scale)
            field_value = getattr(aes, field)
            new_scale.gg_data.col = field_value.col
            new_scale.gg_data.ids = field_value.ids
            setattr(result, field, new_scale)

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
                assign_copy_scale(field_)
    else:
        field = SCALE_TYPE_TO_FIELD.get(scale.scale_type)
        if field is not None:
            assign_copy_scale(field)

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
        col_widths=[
            theme_layout.left,
            Quantity.relative(0.0),
            theme_layout.right
        ],
        row_heights=[
            theme_layout.top,
            Quantity.relative(0.0),
            theme_layout.bottom
        ],
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
    layout = init_theme_margin_layout(
        theme,
        tight_layout,
        requires_legend(filled_scales, theme)
    )
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
                margin=theme_field,
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


def calc_ridge_view_map(ridge: Ridges, label_seq: List[GGValue]) -> Dict[GGValue, int]:
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

    return result


def create_ridge_layout(view: ViewPort, theme: Theme, num_labels: int):
    discr_margin_opt = theme.discrete_scale_margin
    discr_margin = Quantity.relative(0.0)

    if discr_margin_opt is not None:
        discr_margin = discr_margin_opt

    ind_heights = [Quantity.relative(0.0) for _ in range(num_labels)]

    layout(
        view,
        cols=1,
        rows=num_labels + 2,
        row_heights=[discr_margin] + ind_heights + [discr_margin],
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
    # TODO CRITICAL, Medium complexity
    # this calls getYRidgesScale which does not exist
    # if i remember correctly this comes from the macro that does the main/more logic
    # MainAddScales = Tuple[Optional[GGScale], List[GGScale]]
    # main is scale and more is list scale
    # this is very high priority in fixing
    y_ridge_scale = filled_scales.y_ridges[0]
    if y_ridge_scale is None:
        # remove this once the main/more logic is done
        raise GGException("currently only supporting main scale")

    if not isinstance(y_ridge_scale.gg_data.discrete_kind, GGScaleDiscrete):
        raise GGException("expected discrete scale")

    y_label_seq = y_ridge_scale.gg_data.discrete_kind.label_seq
    num_labels = len(y_label_seq)

    y_scale = theme.y_range or filled_scales.y_scale
    y_scale = Scale(low=y_scale.low, high=y_scale.high / ridge.overlap)  # type: ignore
    view.y_scale = y_scale

    view_map = calc_ridge_view_map(ridge, y_label_seq)
    create_ridge_layout(view, theme, num_labels)

    for label, idx in view_map.items():
        view_label = view.children[idx]

        for fg in filled_scales.geoms:
            p_child = view_label.add_viewport_from_coords(
                CoordsInput(), ViewPortInput(name="data")
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


# TODO refactor
def generate_plot(
    view: ViewPort,
    plot: GgPlot,
    filled_scales: FilledScales,
    theme: Theme,
    hide_labels: bool = False,
    hide_ticks: bool = False,
):
    background(view, style=get_plot_background(theme))

    # Change scales to user defined if desired
    view.x_scale = theme.x_range or filled_scales.x_scale

    if plot.ridges is not None:
        ridge = plot.ridges
        generate_ridge(
            view,
            ridge,
            plot,
            filled_scales,
            theme,
            hide_labels,
            hide_ticks
        )
    else:
        view.y_scale = theme.y_range = theme.y_range or filled_scales.y_scale

        for geom in filled_scales.geoms:
            p_child = view.add_viewport_from_coords(
                CoordsInput(), ViewPortInput(name="data")
            )
            create_gobj_from_geom(p_child, geom, theme)
            view.children.append(p_child)

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
        combinations: List[List[GGValue]] = list(
            product([f.gg_data.discrete_kind.label_seq for f in facets])  # type: ignore
        )
    else:
        combinations: List[List[GGValue]] = [[label] for label in facets[0].gg_data.discrete_kind.label_seq]  # type: ignore

    comb_labels: Set[Tuple[str, GGValue]] = set()
    for combination in combinations:
        comb = [(str(fc.gg_data.col), val) for fc, val in zip(facets, combination)]
        for i in comb:
            comb_labels.add(i)

    result: Set[GGValue] = set()

    # TODO critical, this logic is probably a bit off
    # need to get facets working and write unit tests
    for fg in fs.geoms:
        for xk in fg.gg_data.yield_data.keys():
            for _, cb in comb_labels:
                if cb == xk:
                    result.add(cb)

    assert len(result) <= len(combinations)
    return result


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

    if facet.sf_kind in {
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
            facet.sf_kind in {ScaleFreeKind.FREE_X, ScaleFreeKind.FREE}
            or cur_row == rows - 1
            or (cur_row == rows - 2 and cur_col >= last_col and last_col > 0)
        )

        hide_y_labels = not (
            facet.sf_kind in {ScaleFreeKind.FREE_X, ScaleFreeKind.FREE} or cur_col == 0
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
            create_gobj_from_geom(p_child, geom, theme, label_val=label)
            plot_view.children.append(p_child)

        view_label.x_scale = plot_view.x_scale
        view_label.y_scale = plot_view.y_scale

        if not view.x_scale or not view.y_scale:
            # TODO check this
            raise GGException("expected x and y scale")

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


def get_left_bottom(view: ViewPort, annotation: Annotation) -> Tuple[float, float]:
    result_left = 0.0
    result_bottom = 0.0

    if annotation.left is not None:
        result_left = annotation.left
    else:
        if annotation.x is None or view.x_scale is None:
            raise GGException("expected annotation.x and view.x_scale")

        result_left = (
            DataCoordType(
                pos=annotation.x,
                data=DataCoord(axis_kind=AxisKind.X, scale=view.x_scale),
            )
            .to_relative()
            .pos
        )

    if annotation.bottom is not None:
        result_bottom = annotation.bottom
    else:
        if annotation.y is None or view.y_scale is None:
            raise GGException("expected annotation.x and view.x_scale")
        result_left = (
            DataCoordType(
                pos=annotation.y,
                data=DataCoord(axis_kind=AxisKind.X, scale=view.y_scale),
            )
            .to_relative()
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
        left, bottom = get_left_bottom(view, annot)

        margin_h = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=annot.font),
        ).to_relative(length=view.point_height())

        margin_w = StrHeightCoordType(
            pos=ANNOT_RECT_MARGIN,
            data=TextCoordData(text="W", font=annot.font),
        ).to_relative(length=view.point_width())

        total_height = Quantity.relative(
            str_height(annot.text, annot.font)
            .to_relative(length=view.point_height())
            .val
            + margin_h.pos * 2.0,
        )

        font = annot.font
        max_line = list(
            sorted(
                annot.text.split("\n"),
                key=lambda x: get_str_width(x, font).val,
            )
        )[-1]
        max_width = get_str_width(max_line, font)

        rect_width = Quantity.relative(
            max_width.to_relative(length=view.point_width()).val + margin_w.pos * 2.0,
        )

        rect_x = left - margin_w.pos
        rect_y = (
            bottom
            - total_height.to_relative(length=view.point_height()).val
            + margin_h.pos
        )

        annot_rect = None
        if annot.background_color != TRANSPARENT:
            annot_rect = init_rect(
                view,
                origin=Coord(
                    x=RelativeCoordType(pos=rect_x), y=RelativeCoordType(pos=rect_y)
                ),
                width=rect_width,
                height=total_height,
                init_rect_input=InitRectInput(
                    style=rect_style, rotate=annot.rotate, name="annotationBackground"
                ),
            )

        # TODO CRITICAL, easy task
        # double check this logic, make sure its correct
        annot_text = init_multi_line_text(
            view,
            origin=Coord(x=RelativeCoordType(left), y=RelativeCoordType(bottom)),
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


def draw_title(
    view: ViewPort,
    title: str,
    theme: Theme,
    width: Quantity
):
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


def ggcreate(
    plot: GgPlot,
    width: float = 640.0,
    height: float = 480.0
) -> PlotView:
    if len(plot.geoms) == 0:
        raise GGException("Please use at least one `geom`!")

    filled_scales: FilledScales
    if plot.ridges is not None:
        filled_scales = collect_scales(plot.update_aes_ridges())
    else:
        filled_scales = collect_scales(plot)

    theme = build_theme(filled_scales, plot)
    hide_ticks = theme.hide_ticks or False
    hide_labels = theme.hide_labels or False

    img = ViewPort.from_coords(
        CoordsInput(),
        ViewPortInput(
            name="root",
            w_img=Quantity.points(width),
            h_img=Quantity.points(height),
        ),
    )
    background(img, style=get_canvas_background(theme))
    create_layout(img, filled_scales, theme)

    # TODO this isnt very readable
    # maybe img.find_by_name("plot")
    # children generated in plot_layout func
    plt_base = img.children[4]

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

    x_scale = plt_base.x_scale
    y_scale = plt_base.y_scale
    img.x_scale = x_scale
    img.y_scale = y_scale

    img.y_scale = plt_base.y_scale

    drawn_legends: Set[Tuple[DiscreteType, ScaleType]] = set()
    scale_names: Set[str] = set()
    legends: List[ViewPort] = []

    for scale in filled_scales.enumerate_scales_by_id():
        if (
            theme.hide_legend is None
            and scale.scale_type
            not in {ScaleType.LINEAR_DATA, ScaleType.TRANSFORMED_DATA}
            and (scale.gg_data.discrete_kind.discrete_type, scale.scale_type)
            not in drawn_legends
        ):

            # create deep copy of the original legend pane
            lg = deepcopy(img.children[5])
            create_legend(lg, scale, theme.legend_order)

            # TODO low priority, high difficulty
            # support FormulaNode
            scale_col = str(scale.gg_data.col)

            if scale_col not in scale_names:
                legends.append(lg)
                drawn_legends.add(
                    (
                        scale.gg_data.discrete_kind.discrete_type,
                        scale.scale_type
                    )
                )
            scale_names.add(scale_col)

    if len(legends) > 0:
        finalize_legend(img.children[5], legends)
        if plot.theme.legend_position:
            pos = plot.theme.legend_position
            img.children[5].origin.x = pos.x
            img.children[5].origin.y = pos.y

    draw_annotations(img.children[4], plot)

    if plot.title and len(plot.title) > 0:
        draw_title(
            img.children[1],
            plot.title,
            theme,
            img.children[1].point_width().add(img.children[2].point_width()),
        )

    return PlotView(filled_scales=filled_scales, view=img)


def to_tex_options():
    # after alpha version
    raise GGException("unsupported")


def ggmulti(
    plts: List[GgPlot],
    fname: str,
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
            # TODO double check this logic, it goes as float in initViewport
            # our logic is a bit different
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
            # pyright being wrong here, but we should fix anyway
            col_widths=widths_q,  # type: ignore
            row_heights=heights_q,  # type: ignore
        )
    else:
        cols, rows = calc_rows_columns(0, 0, len(plts))
        img = ViewPort.from_coords(
            CoordsInput(),
            # TODO double check this logic, it goes as float in initViewport
            # our logic is a bit different
            ViewPortInput(
                w_img=PointUnit(width * cols),
                h_img=PointUnit(height * rows),
            ),
        )
        layout(img, cols=cols, rows=rows)

    for i, plt in enumerate(plts):
        w_val = widths[i] if i < len(widths) else width
        h_val = heights[i] if i < len(heights) else height

        pp = ggcreate(plt, width=w_val, height=h_val)
        view_embed_at(img, i, pp.view)

    draw_to_file(img, fname)
