import math
from collections import OrderedDict
from dataclasses import field
from typing import Dict, List, Optional, Type, Union

from python_ggplot.colormaps.color_maps import int_to_color
from python_ggplot.common.enum_literals import (
    DISCRETE_TYPE_VALUES,
    SCALE_FREE_KIND_VALUES,
)
from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import (
    GREY92,
    TRANSPARENT,
    WHITE,
    Color,
    Font,
    GGException,
    Gradient,
    LineType,
    Scale,
    TextAlignKind,
    UnitType,
)
from python_ggplot.core.units.objects import Quantity
from python_ggplot.gg.datamancer_pandas_compat import (
    VTODO,
    GGValue,
    VectorCol,
    VFillColor,
    VLinearData,
)
from python_ggplot.gg.scales.base import (
    AlphaScale,
    ColorScale,
    ColorScaleKind,
    FillColorScale,
    GGScale,
    GGScaleContinuous,
    GGScaleData,
    GGScaleDiscrete,
    LinearAndTransformScaleData,
    LinearDataScale,
    ScaleFreeKind,
    ScaleTransformFunc,
    ScaleType,
    ShapeScale,
    SizeScale,
    TransformedDataScale,
)
from python_ggplot.gg.scales.values import (
    FillColorScaleValue,
    ScaleValue,
    SizeScaleValue,
)
from python_ggplot.gg.styles import DEFAULT_COLOR_SCALE
from python_ggplot.gg.types import (
    DataType,
    DiscreteFormat,
    DiscreteType,
    Facet,
    PossibleColor,
    Ridges,
    SecondaryAxis,
    Theme,
)
from python_ggplot.gg.utils import to_opt_sec_axis
from python_ggplot.graphics.draw import layout
from python_ggplot.graphics.initialize import (
    InitRectInput,
    InitTextInput,
    init_coord_1d_from_view,
    init_point_from_coord,
    init_rect,
    init_text,
    init_ticks,
    tick_labels,
)
from python_ggplot.graphics.objects import GOPoint, GraphicsObject
from tests.test_view import (
    AxisKind,
    RelativeCoordType,
    Style,
    TickLabelsInput,
    ViewPort,
)


def ggridges(
    col: str,
    overlap: float = 1.3,
    show_ticks: bool = False,
    label_order: Optional[Dict[GGValue, int]] = None,
) -> Ridges:
    return Ridges(
        col=VectorCol(col),
        overlap=overlap,
        show_ticks=show_ticks,
        label_order=label_order or {},
    )


def facet_wrap(columns: List[str], scale: SCALE_FREE_KIND_VALUES = "fixed") -> Facet:
    scale_ = ScaleFreeKind.eitem(scale)
    facet = Facet(
        columns=columns,
        scale_free_kind=scale_,
    )
    return facet


BASE_TO_LOG = {
    10: math.log10,
    2: math.log2,
}


def _scale_axis_log(
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


def scale_x_log10(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 10
    return _scale_axis_log(AxisKind.X, base, breaks)


def scale_y_log10(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 10
    return _scale_axis_log(AxisKind.Y, base, breaks)


def scale_x_log2(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 2
    return _scale_axis_log(AxisKind.X, base, breaks)


def scale_y_log2(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 2
    return _scale_axis_log(AxisKind.Y, base, breaks)


def sec_axis(
    col: str = "",
    trans_fn: Optional[ScaleTransformFunc] = None,
    inv_trans_fn: Optional[ScaleTransformFunc] = None,
    name: str = "",
) -> SecondaryAxis:

    if trans_fn is not None and inv_trans_fn is not None:
        scale = TransformedDataScale(
            gg_data=GGScaleData.create_empty_scale(col=col),
            transform=trans_fn or TransformedDataScale.defualt_trans,
            inverse_transform=inv_trans_fn
            or TransformedDataScale.defualt_inverse_trans,
        )
        secondary_axis = SecondaryAxis(
            name=name,
            scale=scale,
        )
        return secondary_axis
    elif trans_fn is not None or inv_trans_fn is not None:
        raise GGException(
            "In case of using a transformed secondary scale, both the "
            "forward and reverse transformations have to be provided!"
        )
    else:
        # var fn: Option[FormulaNode]
        # if trans.name.len > 0:
        #   fn = some(trans)
        # do we want to support fornula nodes?
        # so far the answer is either no or not at the moment
        # this may change in the future
        raise GGException("formula nodes are not supported, at least for now")


def _scale_axis_discrete_with_label_fn(
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


def _scale_axis_discrete_with_labels(
    axis_kind: AxisKind,
    name: str = "",
    labels: OrderedDict[GGValue, ScaleValue] = field(default_factory=dict),  # type: ignore
    sec_axis: Optional[SecondaryAxis] = None,
    reversed: bool = False,
) -> GGScale:

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


def scale_x_discrete(
    name: str = "",
    labels_fn: Optional[DiscreteFormat] = None,
    labels: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
) -> GGScale:
    if labels is not None:
        return _scale_axis_discrete_with_labels(AxisKind.X, name, labels, sec_axis)
    else:
        return _scale_axis_discrete_with_label_fn(AxisKind.X, name, labels_fn, sec_axis)


def scale_y_discrete(
    name: str = "",
    labels_fn: Optional[DiscreteFormat] = None,
    labels: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
) -> GGScale:
    if labels is not None:
        return _scale_axis_discrete_with_labels(AxisKind.Y, name, labels, sec_axis)
    else:
        return _scale_axis_discrete_with_label_fn(AxisKind.Y, name, labels_fn, sec_axis)


def _scale_reverse(
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


def scale_x_reverse(
    name: str = "",
    sec_axis: Optional[SecondaryAxis] = None,
    discrete_kind: DISCRETE_TYPE_VALUES = "continuous",
) -> GGScale:
    return _scale_reverse(
        axis_kind=AxisKind.X,
        name=name,
        sec_axis=sec_axis,
        discrete_kind=DiscreteType.eitem(discrete_kind),
    )


def scale_y_reverse(
    name: str = "",
    sec_axis: Optional[SecondaryAxis] = None,
    discrete_kind: DISCRETE_TYPE_VALUES = "continuous",
) -> GGScale:
    return _scale_reverse(
        axis_kind=AxisKind.Y,
        name=name,
        sec_axis=sec_axis,
        discrete_kind=DiscreteType.eitem(discrete_kind),
    )


def scale_fill_continuous(
    name: str = "",
    scale_low: float = 0.0,
    scale_high: float = 0.0,
) -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(name),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(
            data_scale=Scale(low=scale_low, high=scale_high)
        ),
    )
    scale = FillColorScale(
        gg_data=gg_data,
        # TODO does this need a deep copy?
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def scale_fill_discrete(
    name: str = "",
) -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(name),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleDiscrete(),
    )
    scale = FillColorScale(
        gg_data=gg_data,
        # TODO does this need a deep copy?
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def scale_color_continuous(
    name: str = "", scale_low: float = 0.0, scale_high: float = 0.0
) -> GGScale:

    gg_data = GGScaleData(
        col=VectorCol("name"),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(
            data_scale=Scale(low=scale_low, high=scale_high)
        ),
    )
    scale = ColorScaleKind(
        gg_data=gg_data,
        # TODO does this need a deep copy?
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def _scale_color_or_fill_manual(
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
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def scale_fill_manual(values: OrderedDict[GGValue, Color]) -> GGScale:
    return _scale_color_or_fill_manual(FillColorScale, values)


def scale_color_manual(values: OrderedDict[GGValue, Color]) -> GGScale:
    return _scale_color_or_fill_manual(ColorScaleKind, values)


def scale_color_gradient(color_scale: ColorScale | List[int]) -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(""),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(),
    )
    if isinstance(color_scale, ColorScale):
        color_scale_ = color_scale
    else:
        color_scale_ = ColorScale(name="custom", colors=color_scale)

    scale = ColorScaleKind(
        gg_data=gg_data,
        color_scale=color_scale_,
    )
    return scale


def scale_color_identity(col: str = "") -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(col),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(),
    )
    scale = ColorScaleKind(
        gg_data=gg_data,
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def scale_fill_identity(col: str = "") -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(col),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(),
    )
    scale = FillColorScale(
        gg_data=gg_data,
        color_scale=DEFAULT_COLOR_SCALE,
    )
    return scale


def scale_size_identity(col: str = "") -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(col), value_kind=VTODO(), data_type=DataType.SETTING
    )
    scale = SizeScale(
        gg_data=gg_data,
    )
    return scale


def scale_alpha_identity(col: str = "") -> GGScale:
    gg_data = GGScaleData(
        col=VectorCol(col), value_kind=VTODO(), data_type=DataType.SETTING
    )
    scale = AlphaScale(
        gg_data=gg_data,
    )
    return scale


def scale_fill_gradient(color_scale: ColorScale | List[int], name: str = "custom"):
    if isinstance(color_scale, ColorScale):
        color_scale_ = color_scale
    else:
        color_scale_ = ColorScale(name=name, colors=color_scale)

    gg_data = GGScaleData(
        col=VectorCol(""),
        value_kind=VTODO(),
        has_discreteness=True,
        discrete_kind=GGScaleContinuous(),
    )
    scale = ColorScaleKind(
        gg_data=gg_data,
        color_scale=color_scale_,
    )
    return scale


def scale_size_manual(values: OrderedDict[GGValue, float]) -> GGScale:
    # TODO this kind of logic should go somehwere on the discrete object i guess
    # or at least get re-used eventually
    label_seq: List[GGValue] = []
    min_val = float("inf")
    max_val = float("-inf")
    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()

    for k, v in values.items():
        value_map[k] = SizeScaleValue(size=v)
        label_seq.append(k)
        min_val = min(v, min_val)
        max_val = max(v, max_val)

    scale = SizeScale(
        gg_data=GGScaleData(
            col=VectorCol(""),
            value_kind=VTODO(),
            has_discreteness=True,
            discrete_kind=GGScaleDiscrete(
                label_seq=label_seq,
                value_map=value_map,
            ),
        ),
        size_range=(min_val, max_val),
    )

    return scale


def scale_size_discrete(low: float = 2.0, high: float = 7.0) -> GGScale:
    # from  const DefaultSizeRange* = (low: 2.0, high: 7.0)
    # we could set a global one too, keep as is for now

    scale = SizeScale(
        gg_data=GGScaleData(
            col=VectorCol(""),
            value_kind=VTODO(),
            has_discreteness=True,
            discrete_kind=GGScaleDiscrete(),
        ),
        size_range=(low, high),
    )

    return scale


def scale_size_continuous(low: float = 2.0, high: float = 7.0) -> GGScale:
    scale = SizeScale(
        gg_data=GGScaleData(
            col=VectorCol(""),
            value_kind=VTODO(),
            has_discreteness=True,
            discrete_kind=GGScaleContinuous(),
        ),
        size_range=(low, high),
    )

    return scale


def scale_alpha_discrete(low: float = 0.1, high: float = 1.0) -> GGScale:
    # from const DefaultAlphaRange* = (low: 0.1, high: 1.0)
    scale = AlphaScale(
        gg_data=GGScaleData(
            col=VectorCol(""),
            value_kind=VTODO(),
            has_discreteness=True,
            discrete_kind=GGScaleContinuous(),
        ),
        alpha_range=(low, high),
    )

    return scale


def scale_alpha_continuous(low: float = 0.1, high: float = 1.0) -> GGScale:
    # from const DefaultAlphaRange* = (low: 0.1, high: 1.0)
    scale = AlphaScale(
        gg_data=GGScaleData(
            col=VectorCol(""),
            value_kind=VTODO(),
            has_discreteness=True,
            discrete_kind=GGScaleContinuous(),
        ),
        alpha_range=(low, high),
    )

    return scale


def ggtitle(
    title: str,
    sub_title: str = "",
    title_font: Font = field(default_factory=Font),
    sub_title_font: Font = field(default_factory=lambda: Font(size=8.0)),
) -> Theme:
    theme = Theme(
        title=title,
        title_font=title_font,
        sub_title=sub_title or None,
        sub_title_font=sub_title_font,
    )
    return theme


def _generate_legend_markers(
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


def _gen_discrete_legend(
    view: ViewPort, cat: GGScale, access_idx: Optional[List[int]] = None
):
    if not isinstance(cat.gg_data.discrete_kind, GGScaleDiscrete):
        raise GGException("expected a discrete scale")

    markers = _generate_legend_markers(view, cat, access_idx)
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


def _gen_continuous_legend(
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

        markers = _generate_legend_markers(leg_grad, scale, access_idx)
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


def _create_legend(
    view: ViewPort, cat: GGScale, access_idx: Optional[List[int]] = None
) -> None:
    # TODO high priority / easy task
    # double check this to be sure, original code is len(view)
    # i remember ginger sets this up, its either len(view.objects) or  len(view.children)
    start_idx = len(view.children)

    if cat.is_discrete():
        _gen_discrete_legend(view, cat, access_idx)
    elif cat.is_continuous():
        _gen_continuous_legend(view, cat, access_idx)
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


def _finalize_legend(view: ViewPort, legends: List[ViewPort]):
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


def _legend_position(x: float = 0.0, y: float = 0.0):
    return Theme(legend_position=Coord(x=RelativeCoordType(x), y=RelativeCoordType(y)))


def _legend_order(idx: List[int]) -> Theme:
    return Theme(legend_order=idx)


def _hide_legend() -> Theme:
    return Theme(hide_legend=True)


def _canvas_color(color: PossibleColor) -> Theme:
    return Theme(canvas_color=color)


def theme_opaque() -> Theme:
    return Theme(canvas_color=WHITE)


def theme_transparent() -> Theme:
    return Theme(canvas_color=TRANSPARENT)


def theme_void(color: PossibleColor = WHITE) -> Theme:
    return Theme(
        canvas_color=color,
        plot_background_color=color,
        hide_ticks=True,
        hide_tick_labels=True,
        hide_labels=True,
    )
