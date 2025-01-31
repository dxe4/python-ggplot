import math
from ast import Try
from collections import OrderedDict
from dataclasses import field
from typing import Any, Dict, List, Optional, Type, Union

from numpy import _OrderCF

from python_ggplot.common.enum_literals import (
    DISCRETE_TYPE_VALUES,
    SCALE_FREE_KIND_VALUES,
)
from python_ggplot.core.objects import Color, GGException, Scale
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
    Ridges,
    SecondaryAxis,
)
from python_ggplot.gg.utils import to_opt_sec_axis
from tests.test_view import AxisKind


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
