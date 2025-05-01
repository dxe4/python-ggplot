from copy import deepcopy
import os
from collections import OrderedDict
from dataclasses import field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

from python_ggplot.common.enum_literals import (
    DISCRETE_TYPE_VALUES,
    OUTSIDE_RANGE_KIND_VALUES,
    SCALE_FREE_KIND_VALUES,
)
from python_ggplot.core.coord.objects import Coord
from python_ggplot.core.objects import (
    TRANSPARENT,
    WHITE,
    AxisKind,
    Color,
    ColorHCL,
    Font,
    GGException,
    Scale,
    Style,
)
from python_ggplot.core.units.objects import RelativeUnit
from python_ggplot.gg.datamancer_pandas_compat import VTODO, GGValue, VectorCol, VNull
from python_ggplot.gg.scales import ScaleValue, SizeScaleValue
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
    SizeScale,
    TransformedDataScale,
)
from python_ggplot.gg.types import (
    DataType,
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
)
from python_ggplot.graphics.draw import draw_to_file
from python_ggplot.graphics.initialize import InitRectInput, init_rect
from python_ggplot.graphics.views import ViewPort
from python_ggplot.public_interface.utils import (
    ggcreate,
    parse_text_align_string,
    scale_axis_discrete_with_label_fn,
    scale_axis_discrete_with_labels,
    scale_axis_log,
    scale_color_or_fill_manual,
    scale_reverse,
)
from tests.test_view import RelativeCoordType


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


def scale_x_log10(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 10
    return scale_axis_log(AxisKind.X, base, breaks)


def scale_y_log10(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 10
    return scale_axis_log(AxisKind.Y, base, breaks)


def scale_x_log2(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 2
    return scale_axis_log(AxisKind.X, base, breaks)


def scale_y_log2(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 2
    return scale_axis_log(AxisKind.Y, base, breaks)


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


def scale_x_discrete(
    name: str = "",
    labels_fn: Optional[DiscreteFormat] = None,
    labels: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
) -> GGScale:
    if labels is not None:
        return scale_axis_discrete_with_labels(AxisKind.X, name, labels, sec_axis)
    else:
        return scale_axis_discrete_with_label_fn(AxisKind.X, name, labels_fn, sec_axis)


def scale_y_discrete(
    name: str = "",
    labels_fn: Optional[DiscreteFormat] = None,
    labels: Optional[OrderedDict[GGValue, ScaleValue]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
) -> GGScale:
    if labels is not None:
        return scale_axis_discrete_with_labels(AxisKind.Y, name, labels, sec_axis)
    else:
        return scale_axis_discrete_with_label_fn(AxisKind.Y, name, labels_fn, sec_axis)


def _scale_axis_continuous(
    axis_kind: AxisKind,
    breaks: Optional[List[float]] = None,
    labels: Optional[Callable[[float], float]] = None,
    trans: Optional[Callable[[float], float]] = None,
    inv_trans: Optional[Callable[[float], float]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
    name: str = "",
):
    if breaks is None:
        breaks = []

    if sec_axis:
        sec_axis.axis_kind = axis_kind

    if trans is not None and inv_trans is not None:
        transform_data = LinearAndTransformScaleData(
            axis_kind=axis_kind,
            secondary_axis=sec_axis,
        )
        discrete_kind = GGScaleContinuous(
            # todo sanity check this
            format_continuous_label=labels,  # type: ignore
        )
        gg_data = GGScaleData(
            col=VectorCol(name),  # TODO
            value_kind=VNull(),  # TODO
            name=name,
            discrete_kind=discrete_kind,
            has_discreteness=True,
        )
        result = TransformedDataScale(
            gg_data=gg_data,
            data=transform_data,
            transform=trans,
            inverse_transform=inv_trans,
        )

    elif (trans is None) != (inv_trans is None):
        raise Exception(
            "If scale_x_continuous is used for a custom transformed data scale "
            "both the transformation and inverse have to be defined!"
        )
    else:
        transform_data = LinearAndTransformScaleData(
            axis_kind=axis_kind,
            secondary_axis=sec_axis,
        )
        discrete_kind = GGScaleContinuous(
            # todo sanity check this
            format_continuous_label=labels,  # type: ignore
        )
        gg_data = GGScaleData(
            col=VectorCol(name),  # TODO  sanity check
            value_kind=VNull(),  # TODO sanity check
            name=name,
            discrete_kind=discrete_kind,
            has_discreteness=True,
        )
        result = LinearDataScale(
            gg_data=gg_data,
            data=transform_data,
        )

    result.assign_breaks(breaks)
    return result


def scale_x_continuous(
    breaks: Optional[List[float]] = None,
    labels: Optional[Callable[[float], float]] = None,
    trans: Optional[Callable[[float], float]] = None,
    inv_trans: Optional[Callable[[float], float]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
    name: str = "",
):
    result = _scale_axis_continuous(
        axis_kind=AxisKind.X,
        breaks=breaks,
        labels=labels,
        trans=trans,
        inv_trans=inv_trans,
        sec_axis=sec_axis,
        name=name,
    )
    return result


def scale_y_continuous(
    breaks: Optional[List[float]] = None,
    labels: Optional[Callable[[float], float]] = None,
    trans: Optional[Callable[[float], float]] = None,
    inv_trans: Optional[Callable[[float], float]] = None,
    sec_axis: Optional[SecondaryAxis] = None,
    name: str = "",
):
    result = _scale_axis_continuous(
        axis_kind=AxisKind.Y,
        breaks=breaks,
        labels=labels,
        trans=trans,
        inv_trans=inv_trans,
        sec_axis=sec_axis,
        name=name,
    )
    return result


def scale_x_reverse(
    name: str = "",
    sec_axis: Optional[SecondaryAxis] = None,
    discrete_kind: DISCRETE_TYPE_VALUES = "continuous",
) -> GGScale:
    return scale_reverse(
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
    return scale_reverse(
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
        color_scale=ColorScale.viridis(),
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
        color_scale=ColorScale.viridis(),
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
        color_scale=ColorScale.viridis(),
    )
    return scale


def scale_fill_manual(values: OrderedDict[GGValue, Color]) -> GGScale:
    return scale_color_or_fill_manual(FillColorScale, values)


def scale_color_manual(values: OrderedDict[GGValue, Color]) -> GGScale:
    return scale_color_or_fill_manual(ColorScaleKind, values)


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
        color_scale=ColorScale.viridis(),
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
        color_scale=ColorScale.viridis(),
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


def scale_fill_gradient(
    color_scale: Union[ColorScale, List[int]], name: str = "custom"
):
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
    title_font: Optional[Font] = None,
    sub_title_font: Optional[Font] = None,
) -> Theme:

    if title_font is None:
        title_font = Font()

    if sub_title_font is None:
        sub_title_font = Font(size=8.0)

    theme = Theme(
        title=title,
        title_font=title_font,
        sub_title=sub_title or None,
        sub_title_font=sub_title_font,
    )
    return theme


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


def theme_latex() -> Theme:
    return Theme(title_font=Font(size=10.0), label_font=Font(size=10.0))


def prefer_columns() -> Theme:
    return Theme(prefer_rows_over_columns=False)


def prefer_rows() -> Theme:
    return Theme(prefer_rows_over_columns=True)


def xlab(
    label: str = "",
    margin: Optional[float] = None,
    rotate: Optional[float] = None,
    align_to: Literal["none", "left", "right", "center"] = "none",
    tick_margin: Optional[float] = None,
    font_obj: Optional[Font] = None,
    tick_font: Optional[Font] = None,
) -> Theme:
    result = Theme()
    if len(label) > 0:
        result.x_label = label
    if margin is not None:
        result.x_label_margin = margin
    if tick_margin is not None:
        result.x_tick_label_margin = tick_margin
    if rotate is not None:
        result.x_ticks_rotate = rotate
    if font_obj is not None:
        result.label_font = font_obj
    if tick_font is not None:
        result.tick_label_font = tick_font

    x_ticks_text_align = parse_text_align_string(align_to)
    if x_ticks_text_align is not None:
        result.x_ticks_text_align = x_ticks_text_align

    return result


def ylab(
    label: str = "",
    margin: Optional[float] = None,
    rotate: Optional[float] = None,
    align_to: Literal["none", "left", "right", "center"] = "none",
    font_obj: Optional[Font] = None,
    tick_font: Optional[Font] = None,
    tick_margin: Optional[float] = None,
) -> Theme:
    result = Theme()
    if len(label) > 0:
        result.y_label = label
    if margin is not None:
        result.y_label_margin = margin
    if tick_margin is not None:
        result.y_tick_label_margin = tick_margin
    if rotate is not None:
        result.y_ticks_rotate = rotate
    if font_obj is not None:
        result.label_font = font_obj
    if tick_font is not None:
        result.tick_label_font = tick_font
    y_ticks_text_align = parse_text_align_string(align_to)
    if y_ticks_text_align is not None:
        result.y_ticks_text_align = y_ticks_text_align
    return result


def xlim(
    low: Union[int, float],
    high: Union[int, float],
    outside_range: OUTSIDE_RANGE_KIND_VALUES,
) -> Theme:
    or_opt = OutsideRangeKind(outside_range) if outside_range else None
    result = Theme(
        x_range=Scale(low=float(low), high=float(high)),
        x_outside_range=or_opt,
    )
    return result


def ylim(
    low: Union[int, float],
    high: Union[int, float],
    outside_range: OUTSIDE_RANGE_KIND_VALUES,
) -> Theme:
    or_opt = OutsideRangeKind(outside_range) if outside_range else None
    result = Theme(y_range=Scale(float(low), float(high)), y_outside_range=or_opt)
    return result


def ggdraw(view: ViewPort, fname: Union[str, Path]):
    draw_to_file(view, fname)


def ggdraw_plot(plt: PlotView, fname: Union[str, Path]):
    draw_to_file(plt.view, fname)


def ggsave(p: GgPlot, fname: str, width: float = 640.0, height: float = 480.0):
    plt = ggcreate(p, width=width, height=height)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ggdraw(plt.view, fname)


def draw_layout(plt: PlotView, fname: Union[str, Path], filter_view_names: Optional[List[str]] = None):
    base_view = deepcopy(plt.view)
    new_children: List[ViewPort] = []
    colors = ColorHCL.gg_color_hue(len(base_view.children))
    for idx, sub_view in enumerate(base_view.children):
        copied_view = deepcopy(sub_view)
        rect = init_rect(
            copied_view,
            Coord.relative(0.0, 0.0),
            RelativeUnit(1.0),
            RelativeUnit(1.0),
            InitRectInput(style=Style(
                fill_color=colors[idx],
                color=colors[idx]
            ))
        )
        copied_view.children = []
        copied_view.objects = [rect]
        if filter_view_names is None:
            new_children.append(copied_view)
        elif copied_view.name in filter_view_names:
            new_children.append(copied_view)

    base_view.children = new_children

    draw_to_file(base_view, fname)
