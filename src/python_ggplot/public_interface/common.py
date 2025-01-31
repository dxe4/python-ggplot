import math
from typing import Dict, List, Optional, Union

from python_ggplot.common.enum_literals import SCALE_FREE_KIND_VALUES
from python_ggplot.gg.datamancer_pandas_compat import VTODO, GGValue, VectorCol
from python_ggplot.gg.scales.base import (
    GGScale,
    GGScaleContinuous,
    GGScaleData,
    LinearAndTransformScaleData,
    ScaleFreeKind,
    TransformedDataScale,
)
from python_ggplot.gg.types import Facet, Ridges
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
    base = 10
    return _scale_axis_log(AxisKind.X, base, breaks)


def scale_y_log2(breaks: Optional[Union[int, List[float]]] = None) -> GGScale:
    base = 10
    return _scale_axis_log(AxisKind.Y, base, breaks)
