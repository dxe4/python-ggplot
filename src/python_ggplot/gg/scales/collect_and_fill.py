from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_ggplot.gg.utils import GGException
from python_ggplot.colormaps.color_maps import int_to_color
from python_ggplot.core.objects import ColorHCL, LineType, MarkerKind, Scale
from python_ggplot.gg.datamancer_pandas_compat import (  # FormulaType,; ScalarFormula,; pandas_series_to_column,
    FormulaNode,
    GGValue,
    series_is_bool,
    series_is_float,
    series_is_int,
    series_is_obj,
    series_is_str,
    series_value_type,
)
from python_ggplot.gg.scales import (
    AlphaScale,
    GGScale,
    GGScaleDiscrete,
    ScaleValue,
    ShapeScale,
    SizeScale,
)
from python_ggplot.gg.scales.base import GGScaleData, ScaleType, scale_type_to_cls
from python_ggplot.gg.scales.values import (
    ColorScaleValue,
    ShapeScaleValue,
    SizeScaleValue,
    AlphaScaleValue,
    FillColorScaleValue,
)
from python_ggplot.gg.types import DataType, DiscreteType


def add_identity_data(col: "str", df: pd.DataFrame, scale: GGScale):
    # TODO implement when we reach the point where is used
    raise GGException("Not implemented")


def draw_sample_idx(s_high: int, num: int = 100, seed: int = 42) -> NDArray[Any]:
    np.random.seed(seed)
    idx_num = min(num - 1, s_high)
    return np.random.randint(0, s_high + 1, size=idx_num + 1)


def is_discrete_data(
    col: pd.Series[Any],
    s: Scale,
    draw_samples: bool = True,
    discrete_threshold: float = 0.125,
) -> bool:

    if series_is_int(col):
        indices = (
            draw_sample_idx(col.high) if draw_samples else list(range(col.high + 1))
        )
        elements = {col[i] for i in indices}

        if len(elements) > round(len(indices) * discrete_threshold):
            return False
        else:
            return True

    elif series_is_float(col):
        return False

    elif series_is_str(col):
        return True

    elif series_is_bool(col):
        return True

    elif series_is_obj(col):
        indices = (
            draw_sample_idx(col.high) if draw_samples else list(range(col.high + 1))
        )
        # TODO handle pandas object type
        return False

    # TODO test this with multiple data points
    # NaT will probably fail, and there must be other cases
    raise GGException("failed to determine discreteness")


def _is_discrete(
    data: pd.Series[Any], scale: Scale, dc_kind: Optional[DiscreteType] = None
) -> bool:
    if dc_kind is None:
        return is_discrete_data(data, scale, draw_samples=True)
    return dc_kind == DiscreteType.DISCRETE


def discrete_and_type(
    data: pd.Series[Any], scale: Scale, dc_kind: Optional[DiscreteType] = None
) -> Tuple[bool, str]:
    return (_is_discrete(data, scale, dc_kind), series_value_type(data))



def fill_discrete_color_scale(
    scale_kind: ScaleType,
    value_kind: GGValue,
    column: Any,
    data_kind: DataType,
    label_seq: List[Any],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]] = None,
) -> GGScale:
    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(), label_seq=label_seq  # type ignore
    )
    if value_map_opt is not None:
        discrete_kind.value_map = value_map_opt
    else:
        color_cs = ColorHCL.gg_color_hue(len(label_seq))
        for i, k in enumerate(label_seq):
            color=color_cs[i] if data_kind == DataType.MAPPING else int_to_color(k)
            if scale_kind == ScaleType.COLOR:
                discrete_kind.value_map[k] = ColorScaleValue(color=color)
            else:
                discrete_kind.value_map[k] = FillColorScaleValue(
                    color=color
                )

    cls = scale_type_to_cls(scale_kind)
    gg_data = GGScaleData(
        col=column,
        ids=set(),  # type: ignore
        value_kind=value_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    result = cls(gg_data=gg_data)
    return result

def fill_discrete_size_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataType,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    size_range: Tuple[float, float]
) -> GGScale:
    if size_range[0] != size_range[1]:
        raise GGException("Size range must be defined in this context!")

    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()

    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        num_sizes = min(len(label_seq), 5)
        min_size = size_range[0]
        max_size = size_range[1]
        step_size = (max_size - min_size) / float(num_sizes)

        for i, k in enumerate(label_seq):
            if data_kind == DataType.MAPPING:
                size = min_size + float(i) * step_size
            elif data_kind == DataType.SETTING:
                if not isinstance(k, (int, float)):
                    raise GGException("Value used to set size must be Int or Float!")
                size = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = SizeScaleValue(size=size)

    discrete_kind = GGScaleDiscrete(
        value_map=value_map,
        label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    result = SizeScale(
        gg_data=gg_data,
        # TODO high priority why does nim not pass  sizeRange*: tuple[low, high: float]?
        # its passed in the function, but not propagated
        # is this a bug in nim version? if so we should report it
        size_range=size_range,
        size=SizeScaleValue(size=0.0),
    )

    return result


def fill_discrete_alpha_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataType,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    alpha_range: Tuple[float, float]
) -> GGScale:
    # TODO refactor this
    if alpha_range[0] != alpha_range[1]:
        raise GGException("Size range must be defined in this context!")

    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()

    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        num_alphas = len(label_seq)
        min_alpha = alpha_range[0]
        max_alpha = alpha_range[1]
        step_alpha = (max_alpha - min_alpha) / float(num_alphas)

        for i, k in enumerate(label_seq):
            if data_kind == DataType.MAPPING:
                alpha = min_alpha + float(i) * step_alpha
            elif data_kind == DataType.SETTING:
                if not isinstance(k, (int, float)):
                    raise GGException("Value used to set alpha must be Int or Float!")
                alpha = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = AlphaScaleValue(alpha=alpha)

    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(),
        label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )

    # TODO high priority, this 0.0 alpha will cause bugs
    result = AlphaScale(gg_data, alpha=0.0)
    return result


def fill_discrete_shape_scale(
    v_kind: GGValue,
    col: FormulaNode,
    label_seq: List[GGValue],
    value_map_opt:Optional[OrderedDict[GGValue, ScaleValue]] = None,
):
    value_map: OrderedDict[GGValue, ScaleValue] = OrderedDict()
    if value_map_opt is not None:
        value_map = value_map_opt
    else:
        # TODO medium priority i dislike this modulo arithmetic here
        # re-write as a static dict so everyone can read it
        # nice and simple
        for i, k in enumerate(label_seq):
            shape = ShapeScaleValue(
                marker=MarkerKind(i % len(MarkerKind)),
                line_type=LineType((i % (len(LineType) - 1)) + 1),
            )
            value_map[k] = shape

    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(),
        label_seq=label_seq  # type ignore
    )
    gg_data = GGScaleData(
        col=col,
        ids=set(),  # type: ignore
        value_kind=v_kind,
        has_discreteness=True,
        discrete_kind=discrete_kind,
    )
    result = ShapeScale(gg_data=gg_data)
    return result
