from ast import Try
from collections import OrderedDict
from typing import Any, List, Optional, OrderedDict, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gg_ticks import DiscreteType, ScaleType
from gg_utils import GGException
from python_ggplot.colormaps.color_maps import int_to_color
from python_ggplot.core.objects import ColorHCL, Scale
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
    FillColorScaleValue,
    GGScale,
    GGScaleDiscrete,
    ScaleKind,
    ScaleValue,
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
