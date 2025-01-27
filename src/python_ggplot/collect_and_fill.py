from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gg_ticks import DiscreteType, FormulaNode
from gg_utils import GGException
from python_ggplot.colormaps.color_maps import int_to_color
from python_ggplot.core.objects import ColorHCL, LineType, MarkerKind, Scale
from python_ggplot.datamancer_pandas_compat import (  # FormulaType,; ScalarFormula,; pandas_series_to_column,
    GGValue,
    series_is_bool,
    series_is_float,
    series_is_int,
    series_is_obj,
    series_is_str,
    series_value_type,
)
from python_ggplot.gg_scales import (
    AlphaScale,
    AlphaScaleValue,
    FillColorScaleValue,
    GGScale,
    GGScaleDiscrete,
    ScaleKind,
    ScaleValue,
    ShapeScale,
    ShapeScaleValue,
    SizeScale,
    SizeScaleValue,
)
from python_ggplot.gg_types import DataKind


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
    scale_kind: ScaleKind,
    value_kind: GGValue,
    column: Any,
    data_kind: DataKind,
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
            # TODO high priority i think the scale can be one of those types only, since the color is passed in
            # hard code for now, but down the line if this is the case change scale_kind type to be a Union
            # and provide appropriate functions from scale class
            # FillColorScaleValue ColorScaleValue
            scale_value = FillColorScaleValue(
                color=color_cs[i] if data_kind == DataKind.MAPPING else int_to_color(k)
            )
            discrete_kind.value_map[k] = scale_value
    result = GGScale(
        col=column,
        ids=set(),  # type: ignore
        scale_kind=scale_kind,
        value_kind=value_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    return result

def fill_discrete_size_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataKind,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    size_range: Tuple[float, float]
):
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
            if data_kind == DataKind.MAPPING:
                size = min_size + float(i) * step_size
            elif data_kind == DataKind.SETTING:
                assert isinstance(k, (int, float)), "Value used to set size must be Int or Float!"
                size = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = SizeScaleValue(size=size)

    scale_kind = SizeScale()
    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(),
        label_seq=label_seq  # type ignore
    )
    result = GGScale(
        col=col,
        ids=set(),  # type: ignore
        scale_kind=scale_kind,
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )
    return result


def fill_discrete_alpha_scale(
    v_kind: GGValue,
    col: FormulaNode,
    data_kind: DataKind,
    label_seq: List[GGValue],
    value_map_opt: Optional[OrderedDict[GGValue, ScaleValue]],
    alpha_range: Tuple[float, float]
):
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
            if data_kind == DataKind.MAPPING:
                alpha = min_alpha + float(i) * step_alpha
            elif data_kind == DataKind.SETTING:
                if not isinstance(k, (int, float)):
                    raise GGException("Value used to set alpha must be Int or Float!")
                alpha = float(k)
            else:
                raise GGException("unexpected data kind")

            value_map[k] = AlphaScaleValue(alpha=alpha)

    # TODO: CRITICAL Scale kinds Geoms and Discrete types have to be refactored
    # high priority, this 0.0 alpha will cause bugs
    scale_kind = AlphaScale(alpha=0.0)
    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(),
        label_seq=label_seq  # type ignore
    )
    result = GGScale(
        col=col,
        ids=set(),  # type: ignore
        scale_kind=scale_kind,
        value_kind=v_kind,
        has_discreteness=True,
        data_kind=data_kind,
        discrete_kind=discrete_kind,
    )

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

    scale_kind = ShapeScale()
    discrete_kind = GGScaleDiscrete(
        value_map=OrderedDict(),
        label_seq=label_seq  # type ignore
    )
    result = GGScale(
        col=col,
        ids=set(),  # type: ignore
        scale_kind=scale_kind,
        value_kind=v_kind,
        has_discreteness=True,
        discrete_kind=discrete_kind,
    )
    return result
